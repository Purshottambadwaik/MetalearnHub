import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Activity, Layers, Search, Box, Database, Beaker } from 'lucide-react';
import clsx from 'clsx';

// Pages
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import RegisterClass from './pages/RegisterClass';
import Predict from './pages/Predict';
import Dataset from './pages/Dataset';
import Evaluation from './pages/Evaluation';

function Sidebar() {
  const location = useLocation();
  const path = location.pathname;

  const navItems = [
    { path: '/', icon: <LayoutDashboard size={20} />, label: 'Dashboard' },
    { path: '/training', icon: <Activity size={20} />, label: 'Meta-Training' },
    { path: '/register', icon: <Layers size={20} />, label: 'Register Class' },
    { path: '/predict', icon: <Search size={20} />, label: 'Find / Query Image' },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Box size={28} className="text-primary" />
        <div>
          <span className="sidebar-logo block">MetaLearnHub</span>
          <span className="text-[10px] text-muted block -mt-1 font-mono">Few-Shot AI Studio</span>
        </div>
      </div>
      <nav className="nav-links">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={clsx('nav-item', { 'active': path === item.path })}
          >
            {item.icon}
            {item.label}
          </Link>
        ))}

        <div className="pt-6 pb-2 px-4 text-[10px] font-bold text-muted uppercase tracking-wider">
          Experimental
        </div>
        <Link
          to="/evaluation"
          className={clsx('nav-item', { 'active': path === '/evaluation' })}
        >
          <Beaker size={18} />
          <span>Evaluation Lab</span>
        </Link>
        <Link
          to="/dataset"
          className={clsx('nav-item', { 'active': path === '/dataset' })}
        >
          <Database size={18} />
          <span>Raw Data Audit</span>
        </Link>
      </nav>
    </aside>
  );
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <main className="main-content">
          <header className="topbar">
            <h1 className="text-sm font-semibold text-secondary">
              MetaLearnHub <span className="text-text-primary">/ Few-Shot Learning Framework</span>
            </h1>
            <div className="flex items-center gap-4 text-xs text-secondary">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-success inline-block animate-pulse"></span>
                Backend API: <span className="text-success font-semibold">Online (Port 8000)</span>
              </span>
              <span className="hidden sm:inline text-border-light">|</span>
              <span className="hidden sm:inline font-mono text-[11px] bg-surface-elevated px-2 py-1 rounded">
                ResNet-50 Unit-Sphere
              </span>
            </div>
          </header>
          <div className="page-container">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/training" element={<Training />} />
              <Route path="/register" element={<RegisterClass />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/dataset" element={<Dataset />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
