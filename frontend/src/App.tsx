import { useEffect } from 'react';
import { CandidatePopover } from './components/CandidatePopover';
import { Dock } from './components/Dock';
import { Editor } from './components/Editor';
import { RadialWheel } from './components/RadialWheel';
import { Topbar } from './components/Topbar';   // also loads lib/docs (autosave)
import { TreePanel } from './components/TreePanel';
import { engine } from './lib/socket';
import { withDialAxes } from './lib/wheel';
import { useApp } from './store';

export default function App() {
  useEffect(() => {
    engine.onStatus = (conn) => useApp.getState().set({ conn });
    // which moods have programmed dials; make sure each one is on the wheel
    engine.onAxes = (dialAxes) => {
      const { axes, set } = useApp.getState();
      set({ dialAxes, axes: withDialAxes(axes, dialAxes) });
    };
    engine.connect();
    const t = window.setInterval(() => engine.refreshAxes(), 15000);   // pick up newly built dials
    return () => clearInterval(t);
  }, []);
  return (
    <>
      <Topbar />
      <main><Editor /></main>
      <CandidatePopover />
      <RadialWheel />
      <TreePanel />
      <Dock />
    </>
  );
}
