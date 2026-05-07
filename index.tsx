import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import DNNApp from './components/dnn/DNNApp';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Could not find root element to mount to');
}

type AppMode = 'einsum' | 'dnn';
type ThemeMode = 'night' | 'day';

const Root: React.FC = () => {
  const [mode, setMode] = useState<AppMode>('einsum');
  const [theme, setTheme] = useState<ThemeMode>('night');

  return (
    <div className={`h-screen flex flex-col min-h-0 ${theme === 'day' ? 'day-mode' : ''}`}>
      <div className="shrink-0 flex items-center gap-1 px-2 py-1.5 bg-gray-950 border-b border-gray-800">
        <span className="text-[10px] uppercase text-gray-600 font-semibold px-2">Mode</span>
        <div className="flex rounded-md overflow-hidden border border-gray-800">
          <button
            type="button"
            onClick={() => setMode('einsum')}
            className={`px-3 py-1 text-xs font-medium transition-colors ${
              mode === 'einsum'
                ? 'bg-amber-600 text-white'
                : 'bg-gray-900 text-gray-400 hover:text-gray-200'
            }`}
          >
            Tensor contraction
          </button>
          <button
            type="button"
            onClick={() => setMode('dnn')}
            className={`px-3 py-1 text-xs font-medium transition-colors ${
              mode === 'dnn'
                ? 'bg-violet-600 text-white'
                : 'bg-gray-900 text-gray-400 hover:text-gray-200'
            }`}
          >
            DNN network
          </button>
        </div>
        <span className="text-[10px] uppercase text-gray-600 font-semibold px-2 ml-2">Theme</span>
        <div className="flex rounded-md overflow-hidden border border-gray-800">
          <button
            type="button"
            onClick={() => setTheme('night')}
            className={`px-3 py-1 text-xs font-medium transition-colors ${
              theme === 'night'
                ? 'bg-gray-700 text-white'
                : 'bg-gray-900 text-gray-400 hover:text-gray-200'
            }`}
          >
            Night
          </button>
          <button
            type="button"
            onClick={() => setTheme('day')}
            className={`px-3 py-1 text-xs font-medium transition-colors ${
              theme === 'day'
                ? 'bg-sky-600 text-white'
                : 'bg-gray-900 text-gray-400 hover:text-gray-200'
            }`}
          >
            Day
          </button>
        </div>
      </div>
      <div className="flex-1 min-h-0 h-0 flex flex-col overflow-hidden">
        {mode === 'einsum' ? <App /> : <DNNApp />}
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
