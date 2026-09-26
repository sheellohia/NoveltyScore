import { MAX_WORDS } from "../lib/words";

export const WARN_AT = 90;

export default function WordCounter({ count, max = MAX_WORDS }: { count: number; max?: number }) {
  const color =
    count >= max ? "text-red-600 font-semibold" : count >= WARN_AT ? "text-amber-600 font-medium" : "text-slate-500";
  return (
    <span className={`text-xs tabular-nums ${color}`} aria-live="polite">
      {count}/{max} words
    </span>
  );
}
