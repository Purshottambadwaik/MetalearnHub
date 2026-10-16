import { useState, useEffect } from 'react';
import api from '../api';
import { 
  Upload, Plus, Image as ImageIcon, Zap, CheckCircle2, 
  AlertTriangle, Trash2, Database, Sparkles, Sliders
} from 'lucide-react';

export default function Predict() {
  const [classes, setClasses] = useState<any[]>([]);
  const [newClassName, setNewClassName] = useState('');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [collisionNotice, setCollisionNotice] = useState<any | null>(null);
  
  // Custom threshold option
  const [distanceThreshold, setDistanceThreshold] = useState<number>(18.0);
  const [useAutoThreshold, setUseAutoThreshold] = useState<boolean>(true);

  // Prediction state
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string>('');
  const [prediction, setPrediction] = useState<any>(null);

  // Class filter state
  const [filterType, setFilterType] = useState<'all' | 'base' | 'user'>('all');

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = () => {
    api.get('/adaptation/classes').then(res => {
      setClasses(res.data.classes || []);
      setModelLoaded(res.data.model_loaded);
    }).catch(console.error);
  };

  const handleCreateClass = (force: boolean = false) => {
    if (!newClassName.trim()) return;
    setLoading(true);
    setCollisionNotice(null);
    setStatusMessage(null);

    api.post('/adaptation/create-class', { 
      class_name: newClassName.trim(),
      force_new: force 
    })
      .then(res => {
        if (res.data.status === 'already_exists') {
          setCollisionNotice(res.data);
        } else {
          setStatusMessage(`Class '${newClassName}' created successfully!`);
          setNewClassName('');
          fetchClasses();
        }
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const handleSupportUploadBatch = (classId: string, e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    setLoading(true);
    const formData = new FormData();
    for (let i = 0; i < e.target.files.length; i++) {
      formData.append('files', e.target.files[i]);
    }
    
    api.post(`/adaptation/${classId}/add-support-batch`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    .then(() => {
      // Automatically build prototype for convenience
      api.post(`/adaptation/${classId}/build-prototype`).then(() => {
        fetchClasses();
        setLoading(false);
        setCollisionNotice(null);
        setStatusMessage(`Images uploaded and prototype built successfully!`);
      }).catch(() => {
        fetchClasses();
        setLoading(false);
      });
    })
    .catch(err => {
      console.error(err);
      setLoading(false);
    });
  };

  const handleBuildPrototype = (classId: string) => {
    setLoading(true);
    api.post(`/adaptation/${classId}/build-prototype`)
      .then(() => {
        fetchClasses();
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const handleDeleteClass = (classId: string) => {
    if (!confirm('Are you sure you want to delete this class from the system registry?')) return;
    setLoading(true);
    api.delete(`/adaptation/${classId}`)
      .then(() => {
        fetchClasses();
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const handleInitBaseClasses = () => {
    setLoading(true);
    setStatusMessage('Syncing base dataset breeds into registry...');
    api.post('/adaptation/init-base-classes')
      .then(res => {
        setStatusMessage(res.data.message || 'Base classes synced successfully!');
        fetchClasses();
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setStatusMessage('Base dataset directory not found or already synced.');
        setLoading(false);
      });
  };

  const handleQueryUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    setQueryImage(file);
    setQueryPreview(URL.createObjectURL(file));
    setPrediction(null);
  };

  const handlePredict = () => {
    if (!queryImage) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', queryImage);
    
    if (!useAutoThreshold && distanceThreshold > 0) {
      formData.append('distance_threshold', distanceThreshold.toString());
    }
    
    // Default compares against ALL ready classes in system
    const readyClassIds = classes.filter(c => c.is_ready).map(c => c.class_id);
    formData.append('class_ids', JSON.stringify(readyClassIds));
    
    api.post('/prediction', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    .then(res => {
      setPrediction(res.data);
      setLoading(false);
    })
    .catch(err => {
      console.error(err);
      setLoading(false);
    });
  };

  if (!modelLoaded) {
    return (
      <div className="flex flex-col justify-center items-center h-64 text-center">
        <Zap size={48} className="text-warning mb-4" />
        <h2 className="text-xl font-bold mb-2">Inference Engine Initializing...</h2>
        <p className="text-secondary max-w-md">
          Connecting to model checkpoint. Please wait or load a checkpoint from the Evaluation tab.
        </p>
      </div>
    );
  }

  const readyClasses = classes.filter(c => c.is_ready);
  const filteredClasses = classes.filter(cls => {
    if (filterType === 'all') return true;
    return cls.class_type === filterType;
  });

  return (
    <div className="animate-fade-in pb-12">
      <div className="mb-6 flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold mb-1">Persistent Few-Shot Adaptation & Open-Set Inference</h1>
          <p className="text-secondary text-sm">
            Dynamically register new classes into the system without retraining weights. Evaluates query images against both base dataset and user classes, detecting outliers with distance thresholding.
          </p>
        </div>
        <button 
          className="btn btn-secondary text-xs flex items-center gap-2" 
          onClick={handleInitBaseClasses}
          disabled={loading}
          title="Seed base dataset classes into the persistent registry"
        >
          <Database size={14} /> Sync Base Breeds
        </button>
      </div>

      {statusMessage && (
        <div className="mb-4 p-3 bg-primary/10 border border-primary/20 rounded-md text-sm text-primary flex items-center justify-between">
          <span>{statusMessage}</span>
          <button className="text-xs opacity-70 hover:opacity-100" onClick={() => setStatusMessage(null)}>✕</button>
        </div>
      )}

      {collisionNotice && (
        <div className="mb-6 p-4 bg-warning/10 border border-warning/30 rounded-lg text-sm text-warning animate-fade-in">
          <div className="flex items-start gap-3">
            <AlertTriangle size={20} className="shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-bold text-base mb-1">Class Already Exists in System!</h4>
              <p className="mb-3 text-secondary text-sm">
                A class named <strong>"{collisionNotice.class_name}"</strong> is already registered in the persistent system with <strong>{collisionNotice.k_shot} support images</strong>.
              </p>
              <div className="flex items-center gap-3">
                <label className="btn btn-primary text-xs cursor-pointer">
                  <Upload size={14} /> Contribute More Images to "{collisionNotice.class_name}"
                  <input 
                    type="file" 
                    multiple 
                    className="hidden" 
                    accept="image/*" 
                    onChange={(e) => handleSupportUploadBatch(collisionNotice.class_id, e)} 
                    disabled={loading} 
                  />
                </label>
                <button 
                  className="btn btn-secondary text-xs" 
                  onClick={() => setCollisionNotice(null)}
                >
                  Choose Different Name
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-8">
        {/* Left Column: Persistent Class Registry */}
        <div className="flex flex-col gap-6">
          <div className="card">
            <div className="card-header flex justify-between items-center mb-4">
              <h2 className="card-title text-primary flex items-center gap-2">
                <Sparkles size={18} /> 1. System Class Registry
              </h2>
              <div className="flex gap-1 text-xs bg-[var(--bg-main)] p-1 rounded-md border border-[var(--border-light)]">
                <button 
                  className={`px-2 py-1 rounded ${filterType === 'all' ? 'bg-primary text-white font-medium' : 'text-secondary'}`}
                  onClick={() => setFilterType('all')}
                >
                  All ({classes.length})
                </button>
                <button 
                  className={`px-2 py-1 rounded ${filterType === 'base' ? 'bg-primary text-white font-medium' : 'text-secondary'}`}
                  onClick={() => setFilterType('base')}
                >
                  Base ({classes.filter(c => c.class_type === 'base').length})
                </button>
                <button 
                  className={`px-2 py-1 rounded ${filterType === 'user' ? 'bg-primary text-white font-medium' : 'text-secondary'}`}
                  onClick={() => setFilterType('user')}
                >
                  User ({classes.filter(c => c.class_type === 'user').length})
                </button>
              </div>
            </div>
            
            {/* Create Class Input */}
            <div className="flex gap-2 mb-6">
              <input 
                type="text" 
                className="form-control" 
                placeholder="New class name (e.g. Tiger, Hereford)" 
                value={newClassName}
                onChange={e => setNewClassName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreateClass(false)}
              />
              <button 
                className="btn btn-primary whitespace-nowrap" 
                onClick={() => handleCreateClass(false)} 
                disabled={loading || !newClassName.trim()}
              >
                <Plus size={16} /> Register Class
              </button>
            </div>

            {/* Class Cards List */}
            <div className="flex flex-col gap-3 max-h-[500px] overflow-y-auto pr-1">
              {filteredClasses.length === 0 ? (
                <div className="text-center p-6 text-muted border border-dashed border-[var(--border-light)] rounded-md">
                  No classes registered under this filter. Register a new class above or sync base breeds!
                </div>
              ) : (
                filteredClasses.map(cls => (
                  <div 
                    key={cls.class_id} 
                    className={`p-4 rounded-md border transition-all ${
                      cls.is_ready 
                        ? 'border-success/40 bg-[var(--bg-surface)] hover:border-success' 
                        : 'border-[var(--border-light)] bg-[var(--bg-main)]'
                    }`}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-base">{cls.class_name}</h3>
                        {cls.is_ready && <CheckCircle2 size={16} className="text-success" />}
                        <span className={`text-[10px] px-2 py-0.5 rounded font-mono uppercase ${
                          cls.class_type === 'base' ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'
                        }`}>
                          {cls.class_type}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="badge badge-secondary text-xs">{cls.k_shot}-Shot</span>
                        {cls.class_type === 'user' && (
                          <button 
                            className="text-muted hover:text-danger p-1" 
                            title="Delete class"
                            onClick={() => handleDeleteClass(cls.class_id)}
                          >
                            <Trash2 size={14} />
                          </button>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between mt-3 pt-2 border-t border-[var(--border-light)] text-xs text-secondary">
                      <div className="flex items-center gap-2">
                        <label className="btn btn-secondary text-xs cursor-pointer py-1 px-2.5">
                          <Upload size={13} /> Upload Images (Batch)
                          <input 
                            type="file" 
                            multiple 
                            className="hidden" 
                            accept="image/*" 
                            onChange={(e) => handleSupportUploadBatch(cls.class_id, e)} 
                            disabled={loading} 
                          />
                        </label>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {cls.intra_class_radius > 0 && (
                          <span className="text-[11px] text-muted">Radius: {cls.intra_class_radius.toFixed(1)}</span>
                        )}
                        <button 
                          className={`btn text-xs py-1 px-2.5 ${cls.k_shot > 0 && !cls.is_ready ? 'btn-primary' : 'btn-secondary opacity-60'}`}
                          onClick={() => handleBuildPrototype(cls.class_id)}
                          disabled={loading || cls.k_shot === 0 || cls.is_ready}
                        >
                          <Zap size={13} /> {cls.is_ready ? 'Prototype Ready' : 'Build Prototype'}
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Prediction & Open-Set Matching */}
        <div className="flex flex-col gap-6">
          <div className="card">
            <div className="card-header flex justify-between items-center mb-3">
              <h2 className="card-title text-success flex items-center gap-2">
                <Zap size={18} /> 2. Predict Query Image
              </h2>
              <span className="text-xs text-secondary">
                Evaluating against <strong>{readyClasses.length}</strong> active classes
              </span>
            </div>
            
            {readyClasses.length === 0 ? (
              <div className="p-4 text-center text-warning bg-warning/10 rounded-md border border-warning/20">
                Please register at least one class and build its prototype first.
              </div>
            ) : (
              <>
                {/* Distance Threshold Option */}
                <div className="p-3 bg-[var(--bg-main)] rounded-md border border-[var(--border-light)] mb-4 text-xs">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium flex items-center gap-1.5 text-secondary">
                      <Sliders size={13} /> Open-Set Distance Threshold (τ)
                    </span>
                    <button 
                      className={`text-[11px] px-2 py-0.5 rounded ${useAutoThreshold ? 'bg-primary/20 text-primary font-medium' : 'text-muted'}`}
                      onClick={() => setUseAutoThreshold(!useAutoThreshold)}
                    >
                      {useAutoThreshold ? 'Auto (Dynamic)' : 'Manual'}
                    </button>
                  </div>
                  {!useAutoThreshold && (
                    <div className="flex items-center gap-3">
                      <input 
                        type="range" 
                        min="5" 
                        max="35" 
                        step="0.5" 
                        value={distanceThreshold} 
                        onChange={e => setDistanceThreshold(parseFloat(e.target.value))}
                        className="w-full accent-primary" 
                      />
                      <span className="font-mono text-primary font-bold">{distanceThreshold.toFixed(1)}</span>
                    </div>
                  )}
                  <p className="text-muted text-[11px] mt-1">
                    Images exceeding this distance to all prototypes are rejected as <strong>"Not In System / Out-of-Distribution"</strong>.
                  </p>
                </div>

                {/* Upload Query Preview Area */}
                <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-[var(--border-light)] rounded-lg mb-4 bg-[var(--bg-main)] relative overflow-hidden">
                  {queryPreview ? (
                    <img src={queryPreview} alt="Query" className="h-44 object-contain z-10 rounded" />
                  ) : (
                    <div className="flex flex-col items-center text-muted">
                      <ImageIcon size={44} className="mb-2 opacity-40" />
                      <span className="text-sm">Click or drag query image to classify</span>
                    </div>
                  )}
                  
                  <input 
                    type="file" 
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20" 
                    accept="image/*" 
                    onChange={handleQueryUpload} 
                  />
                </div>
                
                <button 
                  className="btn btn-primary w-full py-2.5 text-base font-bold shadow-lg shadow-primary/30 flex items-center justify-center gap-2"
                  onClick={handlePredict}
                  disabled={loading || !queryImage}
                >
                  <Zap size={18} /> Classify & Verify System Match
                </button>
              </>
            )}
          </div>

          {/* Prediction Result Display */}
          {prediction && (
            <div className={`card animate-fade-in border-2 ${
              prediction.is_in_system 
                ? 'border-success/60 bg-gradient-to-br from-[var(--bg-surface)] to-emerald-950/20' 
                : 'border-warning/60 bg-gradient-to-br from-[var(--bg-surface)] to-amber-950/20'
            }`}>
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs uppercase tracking-wider font-semibold text-secondary">
                  Open-Set Verification Result
                </span>
                <span className={`text-xs px-2.5 py-0.5 rounded-full font-bold flex items-center gap-1 ${
                  prediction.is_in_system 
                    ? 'bg-success/20 text-success' 
                    : 'bg-warning/20 text-warning'
                }`}>
                  {prediction.is_in_system ? (
                    <><CheckCircle2 size={13} /> MATCH FOUND</>
                  ) : (
                    <><AlertTriangle size={13} /> NOT IN OUR SYSTEM</>
                  )}
                </span>
              </div>

              <div className="flex items-baseline gap-3 mb-2">
                <div className={`text-2xl font-black ${prediction.is_in_system ? 'text-success' : 'text-warning'}`}>
                  {prediction.predicted_class}
                </div>
                {prediction.is_in_system && prediction.predicted_class_type && (
                  <span className="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary font-mono uppercase">
                    {prediction.predicted_class_type} class
                  </span>
                )}
              </div>

              <p className="text-xs text-secondary mb-4">
                {prediction.status_message}
              </p>

              {/* Metrics Grid */}
              <div className="grid grid-cols-3 gap-2 p-3 bg-[var(--bg-main)] rounded-lg mb-4 text-xs border border-[var(--border-light)]">
                <div>
                  <div className="text-muted text-[11px]">Min Distance</div>
                  <div className="font-mono font-bold text-sm">{prediction.min_distance}</div>
                </div>
                <div>
                  <div className="text-muted text-[11px]">Threshold (τ)</div>
                  <div className="font-mono font-bold text-sm">{prediction.distance_threshold}</div>
                </div>
                <div>
                  <div className="text-muted text-[11px]">Confidence</div>
                  <div className="font-mono font-bold text-sm">{(prediction.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>

              {/* Class Similarities / Distances Ranking */}
              <h4 className="text-xs font-semibold mb-2 text-secondary flex justify-between">
                <span>Ranked Similarities (All Evaluated Classes)</span>
                <span>Distance</span>
              </h4>
              <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
                {prediction.all_classes.map((c: any) => (
                  <div key={c.class_id} className="text-xs p-2 bg-[var(--bg-main)] rounded border border-[var(--border-light)]">
                    <div className="flex justify-between items-center mb-1">
                      <div className="flex items-center gap-1.5 font-medium">
                        <span>{c.class_name}</span>
                        <span className="text-[10px] text-muted font-mono">({c.class_type})</span>
                      </div>
                      <div className="flex items-center gap-2 font-mono">
                        <span className="text-muted">d={c.distance}</span>
                        <span className="font-bold text-primary">{(c.probability * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="w-full bg-[var(--bg-surface)] h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary" 
                        style={{ width: `${c.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="text-[11px] text-muted text-right mt-3 pt-2 border-t border-[var(--border-light)] flex justify-between">
                <span>Evaluated {prediction.total_classes_evaluated} classes in pool</span>
                <span>Inference: {prediction.inference_time_ms}ms</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
