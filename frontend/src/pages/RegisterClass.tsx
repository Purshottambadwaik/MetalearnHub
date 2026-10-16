import React, { useState, useEffect } from 'react';
import api from '../api';
import { 
  Plus, Upload, ShieldCheck, AlertTriangle, CheckCircle2, 
  Trash2, Database, Sparkles, RefreshCw, Layers, Eye, X
} from 'lucide-react';

interface RegisteredClass {
  class_id: string;
  class_name: string;
  class_type: 'base' | 'user';
  description: string;
  k_shot: number;
  is_ready: boolean;
  intra_class_radius: number;
  support_images: string[];
  created_at: string;
}

interface VerificationAnalysis {
  index: number;
  similarity_score: number;
  is_outlier: boolean;
  note: string;
}

interface VerificationResult {
  status: 'passed' | 'warning' | 'rejected';
  batch_coherence: number;
  outlier_count: number;
  image_analysis: VerificationAnalysis[];
  collision_warning: string | null;
  nearest_existing_class: {
    class_name: string;
    class_id: string;
    similarity: number;
    class_type: string;
  } | null;
}

export default function RegisterClass() {
  const [classes, setClasses] = useState<RegisteredClass[]>([]);
  const [loading, setLoading] = useState(false);
  const [filterType, setFilterType] = useState<'all' | 'base' | 'user'>('all');

  // Form State
  const [newClassName, setNewClassName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [filePreviews, setFilePreviews] = useState<string[]>([]);

  // Verification & Quality Gate State
  const [verifying, setVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerificationResult | null>(null);

  // Duplicate / Merge Modal State
  const [duplicateModal, setDuplicateModal] = useState<{
    isOpen: boolean;
    existingClassId: string;
    existingClassName: string;
    existingKShot: number;
  }>({
    isOpen: false,
    existingClassId: '',
    existingClassName: '',
    existingKShot: 0,
  });

  // Contribute to existing class modal state
  const [contributeTarget, setContributeTarget] = useState<RegisteredClass | null>(null);
  const [contributeFiles, setContributeFiles] = useState<File[]>([]);

  const [notification, setNotification] = useState<{
    type: 'success' | 'warning' | 'error';
    message: string;
  } | null>(null);

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = () => {
    setLoading(true);
    api.get('/adaptation/classes')
      .then(res => {
        setClasses(res.data.classes || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching classes:', err);
        setLoading(false);
      });
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const filesArray = Array.from(e.target.files);
    setSelectedFiles(prev => [...prev, ...filesArray]);

    const newPreviews = filesArray.map(f => URL.createObjectURL(f));
    setFilePreviews(prev => [...prev, ...newPreviews]);
    setVerificationResult(null); // Reset previous verification
  };

  const removeSelectedFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    setFilePreviews(prev => prev.filter((_, i) => i !== index));
    setVerificationResult(null);
  };

  // 1. Pre-Flight AI Verification (The "Lion vs Tiger" & Outlier Solution)
  const runVerification = async () => {
    if (selectedFiles.length < 2) {
      setNotification({
        type: 'warning',
        message: 'Please select at least 2 images to evaluate visual coherence and outlier detection.'
      });
      return;
    }

    setVerifying(true);
    const formData = new FormData();
    formData.append('class_name', newClassName || 'New Class');
    selectedFiles.forEach(f => formData.append('files', f));

    try {
      const res = await api.post('/adaptation/verify-batch', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setVerificationResult(res.data);
      if (res.data.status === 'passed') {
        setNotification({
          type: 'success',
          message: `AI Quality Gate Passed! Batch coherence is ${res.data.batch_coherence}%. No outliers detected.`
        });
      } else if (res.data.status === 'warning') {
        setNotification({
          type: 'warning',
          message: res.data.collision_warning || `Warning: ${res.data.outlier_count} visual outlier(s) detected. Check thumbnails below.`
        });
      } else {
        setNotification({
          type: 'error',
          message: 'Batch Coherence Low (< 50%). Uploaded images appear too diverse to represent a single category.'
        });
      }
    } catch (err: any) {
      console.error(err);
      setNotification({
        type: 'error',
        message: 'Failed to complete verification. Ensure backend is running.'
      });
    } finally {
      setVerifying(false);
    }
  };

  // 2. Class Registration with Duplicate Interception (The "Lion" Duplicate Solution)
  const handleRegister = async () => {
    if (!newClassName.trim()) {
      setNotification({ type: 'warning', message: 'Please enter a class name.' });
      return;
    }
    if (selectedFiles.length === 0) {
      setNotification({ type: 'warning', message: 'Please upload at least 3-5 support images for few-shot prototype construction.' });
      return;
    }

    setLoading(true);
    setNotification(null);

    try {
      // Step A: Attempt class creation
      const createRes = await api.post('/adaptation/create-class', {
        class_name: newClassName.trim(),
        description: description.trim(),
        force_new: false,
      });

      // If class already exists, open Duplicate/Merge Modal!
      if (createRes.data.status === 'already_exists') {
        setDuplicateModal({
          isOpen: true,
          existingClassId: createRes.data.class_id,
          existingClassName: createRes.data.class_name,
          existingKShot: createRes.data.k_shot,
        });
        setLoading(false);
        return;
      }

      // Step B: If new, upload support images batch
      const newClassId = createRes.data.class_id;
      const formData = new FormData();
      selectedFiles.forEach(f => formData.append('files', f));

      await api.post(`/adaptation/${newClassId}/add-support-batch`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // Step C: Build prototype
      const protoRes = await api.post(`/adaptation/${newClassId}/build-prototype`);

      setNotification({
        type: 'success',
        message: `Class '${newClassName}' successfully registered! Prototype built from ${protoRes.data.k_shot} support images.`
      });

      // Reset form
      setNewClassName('');
      setDescription('');
      setSelectedFiles([]);
      setFilePreviews([]);
      setVerificationResult(null);
      fetchClasses();
    } catch (err: any) {
      console.error(err);
      setNotification({ type: 'error', message: err.response?.data?.detail || 'Failed to register class.' });
    } finally {
      setLoading(false);
    }
  };

  // 3. Handle Merge & Contribute into Existing Class
  const handleMergeAndContribute = async (targetClassId: string) => {
    if (selectedFiles.length === 0) return;
    setLoading(true);

    try {
      const formData = new FormData();
      selectedFiles.forEach(f => formData.append('files', f));

      const res = await api.post(`/adaptation/${targetClassId}/contribute`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setNotification({
        type: 'success',
        message: `Images successfully merged! Class '${res.data.class_name}' is now upgraded to ${res.data.k_shot}-shot.`
      });

      setDuplicateModal({ isOpen: false, existingClassId: '', existingClassName: '', existingKShot: 0 });
      setNewClassName('');
      setDescription('');
      setSelectedFiles([]);
      setFilePreviews([]);
      setVerificationResult(null);
      fetchClasses();
    } catch (err: any) {
      console.error(err);
      setNotification({ type: 'error', message: 'Failed to contribute images.' });
    } finally {
      setLoading(false);
    }
  };

  // Handle direct contribute modal submission
  const handleDirectContribute = async () => {
    if (!contributeTarget || contributeFiles.length === 0) return;
    setLoading(true);

    try {
      const formData = new FormData();
      contributeFiles.forEach(f => formData.append('files', f));

      const res = await api.post(`/adaptation/${contributeTarget.class_id}/contribute`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setNotification({
        type: 'success',
        message: `Added ${contributeFiles.length} images to '${res.data.class_name}' (Now ${res.data.k_shot}-shot).`
      });

      setContributeTarget(null);
      setContributeFiles([]);
      fetchClasses();
    } catch (err: any) {
      console.error(err);
      setNotification({ type: 'error', message: 'Failed to add images to class.' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClass = async (classId: string, className: string) => {
    if (!window.confirm(`Are you sure you want to remove registered class '${className}'?`)) return;
    try {
      await api.delete(`/adaptation/${classId}`);
      setNotification({ type: 'success', message: `Class '${className}' removed from registry.` });
      fetchClasses();
    } catch (err: any) {
      console.error(err);
      setNotification({ type: 'error', message: 'Failed to delete class.' });
    }
  };

  const filteredClasses = classes.filter(c => {
    if (filterType === 'base') return c.class_type === 'base';
    if (filterType === 'user') return c.class_type === 'user';
    return true;
  });

  return (
    <div className="animate-fade-in pb-12">
      {/* Header */}
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold mb-2 flex items-center gap-2">
            <Layers className="text-primary" /> Class Registration & Quality Gate
          </h1>
          <p className="text-secondary text-sm">
            Dynamically register novel visual categories with few-shot support sets, automated outlier detection, and collaborative class merging.
          </p>
        </div>
        <button 
          onClick={fetchClasses} 
          className="btn btn-secondary flex items-center gap-2 text-sm"
          disabled={loading}
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} /> Refresh Registry
        </button>
      </div>

      {/* Notification Banner */}
      {notification && (
        <div className={`mb-6 p-4 rounded-lg flex items-center justify-between ${
          notification.type === 'success' ? 'bg-success/15 border border-success/30 text-success' :
          notification.type === 'warning' ? 'bg-warning/15 border border-warning/30 text-warning' :
          'bg-danger/15 border border-danger/30 text-danger'
        }`}>
          <div className="flex items-center gap-3">
            {notification.type === 'success' ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
            <span className="text-sm font-medium">{notification.message}</span>
          </div>
          <button onClick={() => setNotification(null)} className="opacity-70 hover:opacity-100">
            <X size={18} />
          </button>
        </div>
      )}

      {/* Main Grid: Registration Form + Quality Gate */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
        {/* Left 2 Cols: Registration & Upload */}
        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Plus className="text-primary" size={20} /> Register New Category
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-secondary mb-1">
                  Class Name (e.g. Lion, Cheetah, Zebra) *
                </label>
                <input
                  type="text"
                  placeholder="Enter category name..."
                  className="input w-full"
                  value={newClassName}
                  onChange={e => setNewClassName(e.target.value)}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-secondary mb-1">
                  Description / Context (Optional)
                </label>
                <input
                  type="text"
                  placeholder="e.g. Wild big cat species from savanna..."
                  className="input w-full"
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                />
              </div>

              {/* Upload Dropzone */}
              <div>
                <label className="block text-sm font-medium text-secondary mb-1">
                  Support Set Images (Upload 5 to 10 representative photos) *
                </label>
                <div className="border-2 border-dashed border-border-light hover:border-primary/50 rounded-xl p-6 text-center bg-surface-elevated/40 transition">
                  <Upload className="mx-auto text-secondary mb-3" size={32} />
                  <p className="text-sm text-primary font-medium mb-1">Click or drag images here to upload</p>
                  <p className="text-xs text-muted mb-4">Supports PNG, JPG, JPEG (Min 3-5 images for solid metric prototype)</p>
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="support-upload"
                  />
                  <label htmlFor="support-upload" className="btn btn-secondary cursor-pointer text-sm">
                    Browse Image Files ({selectedFiles.length} selected)
                  </label>
                </div>
              </div>

              {/* Image Previews & Quality Badges */}
              {filePreviews.length > 0 && (
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-semibold uppercase text-secondary">
                      Selected Support Images ({filePreviews.length})
                    </span>
                    <button 
                      onClick={() => { setSelectedFiles([]); setFilePreviews([]); setVerificationResult(null); }}
                      className="text-xs text-danger hover:underline"
                    >
                      Clear All
                    </button>
                  </div>

                  <div className="grid grid-cols-4 sm:grid-cols-6 gap-3">
                    {filePreviews.map((src, idx) => {
                      const analysis = verificationResult?.image_analysis.find(a => a.index === idx);
                      const isOutlier = analysis?.is_outlier;

                      return (
                        <div 
                          key={idx} 
                          className={`relative group rounded-lg overflow-hidden border aspect-square ${
                            isOutlier ? 'border-danger ring-2 ring-danger/40' : 'border-border-light'
                          }`}
                        >
                          <img src={src} alt={`Support ${idx}`} className="w-full h-full object-cover" />
                          
                          {/* Outlier indicator */}
                          {isOutlier && (
                            <div className="absolute top-1 left-1 bg-danger text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow">
                              Outlier
                            </div>
                          )}

                          {analysis && !isOutlier && (
                            <div className="absolute top-1 left-1 bg-success/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow">
                              {analysis.similarity_score}%
                            </div>
                          )}

                          {/* Delete button */}
                          <button
                            onClick={() => removeSelectedFile(idx)}
                            className="absolute top-1 right-1 p-1 bg-black/70 hover:bg-danger text-white rounded opacity-0 group-hover:opacity-100 transition"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 pt-4 border-t border-border-light">
                <button
                  type="button"
                  onClick={runVerification}
                  disabled={verifying || selectedFiles.length < 2}
                  className="btn btn-secondary flex items-center gap-2 text-sm"
                >
                  <ShieldCheck size={18} className="text-primary" />
                  {verifying ? 'Evaluating Quality...' : 'Run AI Quality Gate'}
                </button>

                <button
                  type="button"
                  onClick={handleRegister}
                  disabled={loading || selectedFiles.length === 0 || !newClassName.trim()}
                  className="btn btn-primary flex-1 flex items-center justify-center gap-2 font-medium"
                >
                  <Sparkles size={18} />
                  {loading ? 'Processing & Building Prototype...' : 'Register Class & Build Prototype'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right 1 Col: AI Quality & Verification Card */}
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-md font-semibold mb-3 flex items-center gap-2">
              <ShieldCheck className="text-success" size={20} /> AI Quality Gate Telemetry
            </h3>
            <p className="text-xs text-secondary mb-4">
              Protects against rogue uploads and mislabeled categories (e.g. uploading Tiger photos under a "Lion" label).
            </p>

            {verificationResult ? (
              <div className="space-y-4">
                <div className="p-3 bg-surface-elevated rounded-lg">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-secondary">Batch Visual Coherence</span>
                    <span className={`text-sm font-bold ${
                      verificationResult.batch_coherence >= 70 ? 'text-success' :
                      verificationResult.batch_coherence >= 50 ? 'text-warning' : 'text-danger'
                    }`}>
                      {verificationResult.batch_coherence}%
                    </span>
                  </div>
                  <div className="w-full bg-border-light rounded-full h-2 overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ${
                        verificationResult.batch_coherence >= 70 ? 'bg-success' :
                        verificationResult.batch_coherence >= 50 ? 'bg-warning' : 'bg-danger'
                      }`}
                      style={{ width: `${verificationResult.batch_coherence}%` }}
                    />
                  </div>
                </div>

                {verificationResult.collision_warning && (
                  <div className="p-3 bg-warning/10 border border-warning/30 rounded-lg text-xs text-warning leading-relaxed flex gap-2">
                    <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                    <span>{verificationResult.collision_warning}</span>
                  </div>
                )}

                {verificationResult.outlier_count > 0 && (
                  <div className="p-3 bg-danger/10 border border-danger/30 rounded-lg text-xs text-danger leading-relaxed flex gap-2">
                    <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                    <span>{verificationResult.outlier_count} image(s) flagged with low visual similarity. Remove them before creating to maintain high model precision.</span>
                  </div>
                )}

                {verificationResult.nearest_existing_class && (
                  <div className="text-xs text-secondary border-t border-border-light pt-3">
                    <span className="block text-muted mb-1">Nearest Existing Class:</span>
                    <span className="font-semibold text-text-primary">
                      {verificationResult.nearest_existing_class.class_name} ({verificationResult.nearest_existing_class.similarity}% similarity)
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-secondary border border-dashed border-border-light rounded-lg">
                <ShieldCheck className="mx-auto text-muted mb-2 opacity-50" size={32} />
                <p className="text-xs">Select support images and click "Run AI Quality Gate" to screen for visual consistency.</p>
              </div>
            )}
          </div>

          <div className="card bg-primary-light/20 border-primary/30">
            <h4 className="text-sm font-semibold text-primary mb-2 flex items-center gap-1.5">
              <Database size={16} /> Few-Shot Meta-Learning Rules
            </h4>
            <ul className="text-xs text-secondary space-y-1.5 list-disc list-inside">
              <li>Requires <strong>3 to 10 images</strong> to calculate the prototype centroid vector.</li>
              <li>Calculates intra-class radius automatically for open-set uncertainty gating.</li>
              <li>Duplicates are merged collaboratively to enrich support density.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Enrolled Classes Gallery */}
      <div>
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Database className="text-primary" size={20} /> Active Registered Classes ({classes.length})
            </h2>
            <p className="text-xs text-secondary">
              All currently enrolled visual concepts available for instant zero-retraining inference.
            </p>
          </div>

          {/* Filter Pills */}
          <div className="flex gap-2">
            {(['all', 'base', 'user'] as const).map(type => (
              <button
                key={type}
                onClick={() => setFilterType(type)}
                className={`px-3 py-1 text-xs rounded-full transition font-medium ${
                  filterType === type 
                    ? 'bg-primary text-white' 
                    : 'bg-surface-elevated text-secondary hover:text-text-primary'
                }`}
              >
                {type === 'all' ? `All (${classes.length})` : 
                 type === 'base' ? `Base (10)` : `User (${classes.length - 10})`}
              </button>
            ))}
          </div>
        </div>

        {/* Classes Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {filteredClasses.map(cls => (
            <div key={cls.class_id} className="card relative group hover:border-primary/50 transition">
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-semibold text-lg text-text-primary">{cls.class_name}</h3>
                <span className={`badge ${
                  cls.class_type === 'base' ? 'badge-primary' : 'badge-secondary text-[#a78bfa] border-[#8b5cf6]/40'
                }`}>
                  {cls.class_type === 'base' ? 'Base Dataset' : 'User Created'}
                </span>
              </div>

              <div className="space-y-1 text-xs text-secondary mb-4">
                <div className="flex justify-between">
                  <span>Support Shots (K):</span>
                  <span className="font-semibold text-text-primary">{cls.k_shot} images</span>
                </div>
                <div className="flex justify-between">
                  <span>Intra-Class Radius:</span>
                  <span className="font-mono text-text-primary">{cls.intra_class_radius.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span className="text-success font-medium flex items-center gap-1">
                    <CheckCircle2 size={12} /> {cls.is_ready ? 'Prototype Ready' : 'Pending'}
                  </span>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex gap-2 pt-3 border-t border-border-light">
                <button
                  onClick={() => setContributeTarget(cls)}
                  className="btn btn-secondary flex-1 text-xs py-1.5 flex items-center justify-center gap-1"
                >
                  <Plus size={14} /> Contribute
                </button>
                {cls.class_type === 'user' && (
                  <button
                    onClick={() => handleDeleteClass(cls.class_id, cls.class_name)}
                    className="p-1.5 text-secondary hover:text-danger hover:bg-danger/10 rounded transition"
                    title="Delete Class"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* MODAL 1: Duplicate Class Interception (The "Lion" Duplicate Solution) */}
      {duplicateModal.isOpen && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 animate-fade-in">
          <div className="card max-w-lg w-full border-warning/40 shadow-2xl">
            <div className="flex items-start gap-3 mb-4">
              <div className="p-2 bg-warning/20 text-warning rounded-lg">
                <AlertTriangle size={24} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-text-primary">
                  Category '{duplicateModal.existingClassName}' Already Exists!
                </h3>
                <p className="text-xs text-secondary mt-1">
                  Another user or the system has already registered this class with <strong>{duplicateModal.existingKShot} support images</strong>.
                </p>
              </div>
            </div>

            <div className="p-4 bg-surface-elevated rounded-lg mb-6 text-sm text-secondary space-y-2">
              <p className="font-medium text-text-primary">Recommended Action: Collaborative Merge</p>
              <p className="text-xs leading-relaxed">
                Your {selectedFiles.length} uploaded images will be added to the existing '{duplicateModal.existingClassName}' class. 
                This upgrades it to <strong>{duplicateModal.existingKShot + selectedFiles.length}-shot</strong>, refining the centroid prototype without overfitting.
              </p>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDuplicateModal({ isOpen: false, existingClassId: '', existingClassName: '', existingKShot: 0 })}
                className="btn btn-secondary text-sm"
              >
                Cancel / Rename
              </button>
              <button
                onClick={() => handleMergeAndContribute(duplicateModal.existingClassId)}
                className="btn btn-primary text-sm flex items-center gap-2"
                disabled={loading}
              >
                <Plus size={16} /> Contribute & Upgrade Class
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 2: Direct Contribute More Images */}
      {contributeTarget && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 animate-fade-in">
          <div className="card max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-text-primary">
                Contribute Images to '{contributeTarget.class_name}'
              </h3>
              <button onClick={() => setContributeTarget(null)} className="text-secondary hover:text-text-primary">
                <X size={18} />
              </button>
            </div>

            <p className="text-xs text-secondary mb-4">
              Currently has {contributeTarget.k_shot} support images. Upload additional images to expand its metric variance.
            </p>

            <div className="mb-4">
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={e => setContributeFiles(Array.from(e.target.files || []))}
                className="input w-full text-xs"
              />
              <span className="text-xs text-muted mt-1 block">
                {contributeFiles.length} new files selected.
              </span>
            </div>

            <div className="flex justify-end gap-3">
              <button onClick={() => setContributeTarget(null)} className="btn btn-secondary text-sm">
                Cancel
              </button>
              <button
                onClick={handleDirectContribute}
                disabled={loading || contributeFiles.length === 0}
                className="btn btn-primary text-sm flex items-center gap-2"
              >
                <Upload size={16} /> Upload & Recalculate Prototype
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
