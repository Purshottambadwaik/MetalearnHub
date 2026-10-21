import React, { useState, useEffect } from 'react';
import api from '../api';
import { 
  Search, Upload, Sparkles, CheckCircle2, AlertTriangle, 
  Sliders, ShieldAlert, Cpu, BarChart2, Layers
} from 'lucide-react';

interface PredictionCandidate {
  class_id: string;
  class_name: string;
  class_type: string;
  distance: number;
  confidence: number;
  k_shot?: number;
  intra_class_radius?: number;
}

interface PredictionResponse {
  predicted_class: string;
  class_id: string | null;
  confidence: number;
  raw_confidence: number;
  distance: number;
  open_set_status: string;
  open_set_accepted: boolean;
  rejection_reason?: string;
  candidates: PredictionCandidate[];
  algorithm_used: string;
  total_registered_classes: number;
}

export default function Predict() {
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string>('');
  const [algorithm, setAlgorithm] = useState<string>('ame_net');
  const [distanceThreshold, setDistanceThreshold] = useState<number>(50.0);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [classesCount, setClassesCount] = useState<number>(0);

  useEffect(() => {
    api.get('/adaptation/classes').then(res => {
      setClassesCount(res.data.classes?.length || 0);
    }).catch(console.error);
  }, []);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setQueryImage(file);
      setQueryPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!queryImage) {
      setError('Please upload a query image to classify.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', queryImage);
    formData.append('algorithm', algorithm);
    // Convert 0-100 threshold slider to decimal 0.0-1.0
    formData.append('distance_threshold', (distanceThreshold / 100.0).toString());

    try {
      const res = await api.post('/prediction', formData);
      setPrediction(res.data);
    } catch (err: any) {
      console.error(err);
      const detail = err.response?.data?.detail || err.response?.data?.message || err.message;
      setError(detail || 'Prediction failed. Ensure the backend server is active.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in pb-12">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Search className="text-primary" /> Query Image Recognition Studio
        </h1>
        <p className="text-secondary text-sm">
          Classify any query image across all {classesCount} enrolled visual categories in real time with Open-Set Out-of-Distribution rejection.
        </p>
      </div>

      {/* Main Grid: Controls + Upload + Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Upload & Inference Parameters (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="card">
            <h2 className="text-md font-semibold mb-4 flex items-center gap-2">
              <Upload size={18} className="text-primary" /> 1. Upload Query Image
            </h2>

            {/* Dropzone */}
            <div className="border-2 border-dashed border-border-light hover:border-primary/50 rounded-xl p-6 text-center bg-surface-elevated/40 transition">
              {queryPreview ? (
                <div className="relative group mx-auto w-48 h-48 rounded-lg overflow-hidden border border-border-light shadow-md mb-4">
                  <img src={queryPreview} alt="Query preview" className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                    <label htmlFor="query-upload" className="btn btn-secondary text-xs cursor-pointer">
                      Change Photo
                    </label>
                  </div>
                </div>
              ) : (
                <div className="py-6">
                  <Search className="mx-auto text-secondary mb-3 opacity-60" size={36} />
                  <p className="text-sm font-medium text-primary mb-1">Click or drag a test image here</p>
                  <p className="text-xs text-muted mb-4">PNG, JPG, WebP supported</p>
                </div>
              )}

              <input
                type="file"
                accept="image/*"
                id="query-upload"
                onChange={handleImageSelect}
                className="hidden"
              />
              {!queryPreview && (
                <label htmlFor="query-upload" className="btn btn-secondary text-sm cursor-pointer inline-block">
                  Browse File
                </label>
              )}
            </div>
          </div>

          {/* Inference Settings */}
          <div className="card">
            <h3 className="text-md font-semibold mb-4 flex items-center gap-2">
              <Sliders size={18} className="text-primary" /> 2. Algorithm & Threshold Settings
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold uppercase text-secondary mb-1">
                  Active Algorithm
                </label>
                <select
                  value={algorithm}
                  onChange={e => setAlgorithm(e.target.value)}
                  className="input w-full text-sm"
                >
                  <option value="ame_net">AME-Net: Adaptive Metric-Exemplar Hybrid (Recommended)</option>
                  <option value="ensemble">4-Way Consensus Ensemble (Voting Fusion)</option>
                  <option value="protonet">Prototypical Networks (Metric Centroid)</option>
                  <option value="matching_net">Matching Networks (Exemplar Attention)</option>
                  <option value="relation_net">Relation Networks (Neural Comparator)</option>
                  <option value="maml">MAML (First-Order Gradient Adapt)</option>
                </select>
                <span className="text-[11px] text-muted mt-1 block">
                  AME-Net dynamically weights centroid clustering vs exemplar memory using support set dispersion to beat individual models.
                </span>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs font-semibold uppercase text-secondary">
                    Certainty Gate (Rejection Threshold)
                  </label>
                  <span className="font-mono text-xs font-bold text-primary">{distanceThreshold}%</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="90"
                  step="5"
                  value={distanceThreshold}
                  onChange={e => setDistanceThreshold(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-muted mt-1">
                  <span>Permissive (20%)</span>
                  <span>Balanced (50%)</span>
                  <span>Strict (90%)</span>
                </div>
              </div>

              <button
                type="button"
                onClick={handlePredict}
                disabled={loading || !queryImage}
                className="btn btn-primary w-full py-2.5 flex items-center justify-center gap-2 font-medium text-sm mt-4"
              >
                <Sparkles size={18} />
                {loading ? 'Running Few-Shot Recognition...' : 'Classify Query Image'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results & Explanation (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {error && (
            <div className="card border-danger/40 bg-danger/10 text-danger p-4 flex items-center gap-3 text-sm">
              <AlertTriangle size={20} className="shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {prediction ? (
            <div className="space-y-6 animate-fade-in">
              {/* Main Classification Result Banner */}
              <div className={`card border-2 ${
                prediction.open_set_accepted 
                  ? 'border-success/50 bg-success/5' 
                  : 'border-danger/50 bg-danger/5'
              }`}>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-secondary block mb-1">
                      System Verdict
                    </span>
                    <h2 className="text-3xl font-extrabold text-text-primary flex items-center gap-2">
                      {prediction.open_set_accepted ? (
                        <>
                          <CheckCircle2 className="text-success" size={28} />
                          {prediction.predicted_class}
                        </>
                      ) : (
                        <>
                          <ShieldAlert className="text-danger" size={28} />
                          [OUT-OF-DISTRIBUTION / REJECTED]
                        </>
                      )}
                    </h2>
                  </div>

                  <div className="text-right">
                    <span className="text-xs text-secondary block mb-1">Confidence</span>
                    <span className={`text-2xl font-mono font-bold ${
                      prediction.open_set_accepted ? 'text-success' : 'text-danger'
                    }`}>
                      {prediction.confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Explanation text */}
                <p className="text-xs leading-relaxed text-secondary border-t border-border-light pt-3">
                  {prediction.open_set_accepted ? (
                    <>
                      Query matched registered category <strong>'{prediction.predicted_class}'</strong> with {prediction.confidence.toFixed(1)}% confidence, satisfying the {distanceThreshold}% certainty gate and intra-class dispersion bounds.
                    </>
                  ) : (
                    <>
                      <strong>Out-of-Distribution Warning:</strong> {prediction.rejection_reason || 'Query does not sufficiently match any enrolled concept. Certainty threshold set to 0.0% to prevent closed-set hallucination.'}
                    </>
                  )}
                </p>
              </div>

              {/* Candidate Similarity Distribution */}
              <div className="card">
                <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
                  <BarChart2 size={16} className="text-primary" /> Candidate Similarity Distribution
                </h3>

                <div className="space-y-3">
                  {prediction.candidates.slice(0, 5).map((cand, idx) => (
                    <div key={cand.class_id || idx} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="font-medium text-text-primary flex items-center gap-1.5">
                          <span className="text-muted font-mono">{idx + 1}.</span>
                          {cand.class_name}
                          <span className={`text-[10px] px-1.5 py-0.2 rounded ${
                            cand.class_type === 'base' ? 'bg-primary/20 text-primary' : 'bg-[#8b5cf6]/20 text-[#a78bfa]'
                          }`}>
                            {cand.class_type}
                          </span>
                        </span>
                        <span className="font-mono text-secondary">
                          {cand.confidence.toFixed(1)}% (dist: {cand.distance.toFixed(3)})
                        </span>
                      </div>
                      <div className="w-full bg-border-light rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            idx === 0 && prediction.open_set_accepted ? 'bg-success' : 'bg-primary/60'
                          }`}
                          style={{ width: `${Math.min(100, Math.max(0, cand.confidence))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Algorithm Telemetry Details */}
              <div className="card bg-surface-elevated/40 text-xs space-y-2">
                <div className="flex justify-between text-secondary">
                  <span>Evaluated By Algorithm:</span>
                  <span className="font-semibold text-text-primary uppercase font-mono">{prediction.algorithm_used}</span>
                </div>
                <div className="flex justify-between text-secondary">
                  <span>Enrolled Search Space:</span>
                  <span className="font-semibold text-text-primary">{prediction.total_registered_classes} Classes</span>
                </div>
                <div className="flex justify-between text-secondary">
                  <span>Nearest Prototype Distance:</span>
                  <span className="font-mono text-text-primary">{prediction.distance.toFixed(4)}</span>
                </div>
              </div>
            </div>
          ) : (
            /* Idle Placeholder */
            <div className="card text-center py-16 border-dashed border-border-light">
              <Cpu className="mx-auto text-muted mb-3 opacity-40" size={48} />
              <h3 className="text-md font-semibold text-text-primary mb-1">Awaiting Query Input</h3>
              <p className="text-xs text-secondary max-w-sm mx-auto">
                Upload a test image on the left and click "Classify Query Image" to inspect real-time feature projections and open-set rejection.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
