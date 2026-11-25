import { useCallback, useState } from 'react';

function saveTheme(value) {
  try { localStorage.setItem('theme', value); } catch { /* storage can be blocked; the choice just won't persist */ }
}

// The early script in index.html sets data-theme from the saved choice; otherwise follow the system.
export function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set === 'dark' || set === 'light') return set;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function useTheme() {
  const [theme, setTheme] = useState(currentTheme);
  const toggle = useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    setTheme(next);
    saveTheme(next);
  }, [theme]);
  return [theme, toggle];
}
