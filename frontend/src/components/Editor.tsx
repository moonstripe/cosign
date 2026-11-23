import { useEffect, useRef } from 'react';
import { editor } from '../lib/editorController';
import { useApp } from '../store';

export function Editor() {
  const ref = useRef<HTMLDivElement>(null);
  const blank = useApp((s) => s.docText === '');
  useEffect(() => {
    editor.attach(ref.current!);
    return () => editor.detach();
  }, []);
  return (
    <div
      id="ed"
      ref={ref}
      className={blank ? 'blank' : undefined}
      contentEditable
      suppressContentEditableWarning
      spellCheck={false}
      data-placeholder="Start writing. Press ⌥Tab when you want the model to write."
    />
  );
}
