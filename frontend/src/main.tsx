import { createRoot } from 'react-dom/client';
import App from './App';
// type system from moonstripe_cybernetics: Mona Sans (UI), Monaspace Neon (mono), Geist Pixel (wordmark)
import '@fontsource-variable/mona-sans/standard.css';
import '@fontsource/monaspace-neon/400.css';
import '@fontsource/monaspace-neon/700.css';
import './index.css';

// No StrictMode: the editor and the animation loop are imperative, and double
// mounting would attach the editor twice.
createRoot(document.getElementById('root')!).render(<App />);

// dev only: poke the live app from the console / tests (window.__cowrite.useApp.getState())
if (import.meta.env.DEV) {
  Promise.all([import('./store'), import('./lib/editorController'), import('./lib/wheel'), import('./lib/docs')]).then(([s, e, w, d]) => {
    (window as unknown as { __cowrite: object }).__cowrite = { useApp: s.useApp, editor: e.editor, wheel: w, docs: d };
  });
}
