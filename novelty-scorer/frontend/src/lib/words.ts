// Must match backend/models.py count_words (whitespace-delimited).

export const MAX_WORDS = 100;

export function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

/** Cut text right after its `max`-th word, keeping original spacing. */
export function truncateWords(text: string, max: number): string {
  const re = /\S+/g;
  let n = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    n += 1;
    if (n === max) return text.slice(0, m.index + m[0].length);
  }
  return text;
}
