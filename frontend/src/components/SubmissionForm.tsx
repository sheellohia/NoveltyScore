import { useState, type FormEvent } from "react";
import WordCounter, { WARN_AT } from "./WordCounter";
import { MAX_WORDS, countWords, truncateWords } from "../lib/words";

const HEADER_SOFT_CAP = 120;
const TAKEAWAY_MAX = 120; // hard limit, must match backend MAX_TAKEAWAY_CHARS

export interface SubmissionValues {
  header: string;
  content: string;
  takeaway: string;
}

interface Props {
  submitting: boolean;
  onSubmit: (values: SubmissionValues) => void;
}

const inputCls =
  "w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100";

export default function SubmissionForm({ submitting, onSubmit }: Props) {
  const [header, setHeader] = useState("");
  const [content, setContent] = useState("");
  const [takeaway, setTakeaway] = useState("");
  const [trimmed, setTrimmed] = useState(false);

  const words = countWords(content);

  // Hard cap: never let content exceed MAX_WORDS. Deletions always allowed;
  // an over-limit paste is trimmed to the first MAX_WORDS words.
  function handleContentChange(next: string) {
    if (countWords(next) <= MAX_WORDS) {
      setContent(next);
      setTrimmed(false);
      return;
    }
    if (next.length < content.length) {
      setContent(next);
      return;
    }
    const cut = truncateWords(next, MAX_WORDS);
    if (cut.length > content.length) {
      // pasted in a big chunk: keep what fits
      setContent(cut);
      setTrimmed(true);
    } else {
      // typing a 101st word: block the keystroke
      setTrimmed(false);
    }
  }

  const valid =
    header.trim() !== "" &&
    takeaway.trim() !== "" &&
    takeaway.length <= TAKEAWAY_MAX &&
    words > 0 &&
    words <= MAX_WORDS;

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!valid || submitting) return;
    onSubmit({ header: header.trim(), content: content.trim(), takeaway: takeaway.trim() });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Your submission</h2>

      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <label htmlFor="header" className="text-sm font-medium">
            Header
          </label>
          <span
            className={`text-xs tabular-nums ${header.length > HEADER_SOFT_CAP ? "text-amber-600" : "text-slate-400"}`}
          >
            {header.length}/{HEADER_SOFT_CAP}
          </span>
        </div>
        <input
          id="header"
          className={inputCls}
          value={header}
          onChange={(e) => setHeader(e.target.value)}
          placeholder="A one-line headline for your submission"
        />
        {header.length > HEADER_SOFT_CAP && (
          <p className="mt-1 text-xs text-amber-600">Headers work best under {HEADER_SOFT_CAP} characters.</p>
        )}
      </div>

      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <label htmlFor="content" className="text-sm font-medium">
            Content
          </label>
          <WordCounter count={words} />
        </div>
        <textarea
          id="content"
          rows={6}
          className={`${inputCls} resize-y ${
            words >= MAX_WORDS
              ? "border-red-400 focus:border-red-500 focus:ring-red-100"
              : words >= WARN_AT
                ? "border-amber-400 focus:border-amber-500 focus:ring-amber-100"
                : ""
          }`}
          value={content}
          onChange={(e) => handleContentChange(e.target.value)}
          placeholder="Your paragraph (max 100 words)"
        />
        {words >= MAX_WORDS && (
          <p className="mt-1 text-xs text-red-600">
            {trimmed ? "Pasted text was trimmed to 100 words." : "100-word limit reached."}
          </p>
        )}
      </div>

      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <label htmlFor="takeaway" className="text-sm font-medium">
            Takeaway
          </label>
          <span
            className={`text-xs tabular-nums ${
              takeaway.length >= TAKEAWAY_MAX
                ? "font-semibold text-red-600"
                : takeaway.length >= TAKEAWAY_MAX * 0.9
                  ? "font-medium text-amber-600"
                  : "text-slate-400"
            }`}
          >
            {takeaway.length}/{TAKEAWAY_MAX}
          </span>
        </div>
        <input
          id="takeaway"
          className={`${inputCls} ${takeaway.length >= TAKEAWAY_MAX ? "border-red-400 focus:border-red-500 focus:ring-red-100" : ""}`}
          value={takeaway}
          maxLength={TAKEAWAY_MAX}
          onChange={(e) => setTakeaway(e.target.value.slice(0, TAKEAWAY_MAX))}
          placeholder="The one thing a reader should remember"
        />
        {takeaway.length >= TAKEAWAY_MAX && (
          <p className="mt-1 text-xs text-red-600">{TAKEAWAY_MAX}-character limit reached.</p>
        )}
      </div>

      <button
        type="submit"
        disabled={!valid || submitting}
        className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {submitting && <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />}
        {submitting ? "Scoring…" : "Score my submission"}
      </button>
    </form>
  );
}
