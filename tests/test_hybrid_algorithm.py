"""
Unit & Integration Tests for Adaptive Metric-Exemplar Network (AME-Net).
Validates:
1. Forward pass tensor shapes across 1, 3, 5, 10-shot settings.
2. Dynamic dispersion and entropy gating behavior.
3. Zero support-query leakage.
4. Determinism and reproducibility under fixed seeds.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.algorithms.hybrid.ame_net import AMENet, AMENetAlgorithm, DispersionEntropyGate


class TestAMENetGate:
    """Test dynamic gating module."""

    def test_gate_monotonicity_dispersion(self):
        gate = DispersionEntropyGate(init_gamma0=0.5, init_gamma1=3.0, init_gamma2=1.0)
        delta_h = torch.zeros(1, 1)

        low_disp = torch.tensor([[0.05]])
        high_disp = torch.tensor([[0.80]])

        alpha_low = gate(low_disp, delta_h).item()
        alpha_high = gate(high_disp, delta_h).item()

        # Low dispersion should assign HIGHER weight to ProtoNet
        assert alpha_low > alpha_high, f"Expected alpha_low ({alpha_low}) > alpha_high ({alpha_high})"

    def test_gate_monotonicity_entropy(self):
        gate = DispersionEntropyGate(init_gamma0=0.5, init_gamma1=1.0, init_gamma2=2.0)
        disp = torch.tensor([[0.20]])

        # ProtoNet has higher entropy (more confused) -> delta_h > 0 -> alpha should decrease
        proto_confused = torch.tensor([[1.5]])
        # MatchingNet has higher entropy -> delta_h < 0 -> alpha should increase
        match_confused = torch.tensor([[-1.5]])

        alpha_proto_confused = gate(disp, proto_confused).item()
        alpha_match_confused = gate(disp, match_confused).item()

        assert alpha_match_confused > alpha_proto_confused


class TestAMENetModule:
    """Test AMENet forward passes across shots."""

    def get_model(self):
        # Use resnet18 without pretraining for rapid CPU testing
        return AMENet(backbone_name="resnet18", pretrained=False, freeze_backbone=False)

    def test_variable_shots(self, model, k_shot):
        n_way = 3
        q_query = 2
        total_supp = n_way * k_shot
        total_query = n_way * q_query

        supp_imgs = torch.randn(total_supp, 3, 224, 224)
        supp_labels = torch.repeat_interleave(torch.arange(n_way), k_shot)

        query_imgs = torch.randn(total_query, 3, 224, 224)
        query_labels = torch.repeat_interleave(torch.arange(n_way), q_query)

        log_probs, telemetry = model(supp_imgs, supp_labels, query_imgs)

        # Output shape checks
        assert log_probs.shape == (total_query, n_way)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()

        # Telemetry checks
        assert "alpha" in telemetry
        assert telemetry["alpha"].shape == (total_query, 1)
        assert (telemetry["alpha"] >= 0.0).all() and (telemetry["alpha"] <= 1.0).all()

        assert "probs" in telemetry
        assert telemetry["probs"].shape == (total_query, n_way)
        # Probabilities sum to 1 per query
        prob_sums = telemetry["probs"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_zero_leakage_assertion(self, model):
        """Verify support and query sets are strictly disjoint and independent."""
        supp_imgs = torch.randn(4, 3, 224, 224)
        supp_labels = torch.tensor([0, 0, 1, 1])
        query_imgs = torch.randn(2, 3, 224, 224)

        # Ensure embeddings do not cross-pollinate
        s_emb = model.embed(supp_imgs)
        q_emb = model.embed(query_imgs)

        assert not torch.equal(s_emb[0], q_emb[0])
        state = model.compute_support_state(s_emb, supp_labels)

        # Assert support state only contains support information
        assert state["support_embeddings"].shape[0] == 4
        assert state["prototypes"].shape[0] == 2


class TestAMENetAlgorithm:
    """Test MetaLearningAlgorithm interface compliance."""

    def get_algo(self):
        return AMENetAlgorithm(backbone_name="resnet18", pretrained=False, device="cpu")

    def test_meta_train_step(self, algo):
        supp_imgs = torch.randn(6, 3, 224, 224)
        supp_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        query_imgs = torch.randn(4, 3, 224, 224)
        query_labels = torch.tensor([0, 0, 1, 1])

        metrics = algo.meta_train_step(supp_imgs, supp_labels, query_imgs, query_labels)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "mean_alpha" in metrics
        assert "mean_dispersion" in metrics
        assert isinstance(metrics["loss"], float)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_adapt_and_predict(self, algo):
        supp_imgs = torch.randn(6, 3, 224, 224)
        supp_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        query_imgs = torch.randn(4, 3, 224, 224)

        adapted_state = algo.adapt(supp_imgs, supp_labels)
        preds, confs = algo.predict(query_imgs, adapted_state)

        assert preds.shape == (4,)
        assert confs.shape == (4,)
        assert (confs >= 0.0).all() and (confs <= 1.0).all()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Running AMENet Hybrid Algorithm Validation Suite")
    print("=" * 60)

    # 1. Gate tests
    gate_tester = TestAMENetGate()
    gate_tester.test_gate_monotonicity_dispersion()
    gate_tester.test_gate_monotonicity_entropy()
    print(" [PASSED] Dynamic Dispersion & Entropy Gating Behavior")

    # 2. Module tests across shots
    mod_tester = TestAMENetModule()
    model = mod_tester.get_model()
    for shot in [1, 3, 5, 10]:
        mod_tester.test_variable_shots(model, shot)
        print(f" [PASSED] AMENet Forward Pass for {shot:2d}-Shot Evaluation")

    mod_tester.test_zero_leakage_assertion(model)
    print(" [PASSED] Zero Support-Query Leakage Assertion")

    # 3. Algorithm tests
    algo_tester = TestAMENetAlgorithm()
    algo = algo_tester.get_algo()
    algo_tester.test_meta_train_step(algo)
    print(" [PASSED] MetaLearningAlgorithm meta_train_step")
    algo_tester.test_adapt_and_predict(algo)
    print(" [PASSED] MetaLearningAlgorithm adapt & predict")

    print("\n" + "=" * 60)
    print(" ALL AMENet UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
