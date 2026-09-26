import { useEffect, useRef } from "react";
import { NavLink, useLocation } from "react-router-dom";
import type { TopicSummary } from "../lib/types";

export default function TopicNav({ topics }: { topics: TopicSummary[] }) {
  const navRef = useRef<HTMLElement>(null);
  const { pathname } = useLocation();

  // keep the active tab visible when the nav scrolls horizontally (narrow screens)
  useEffect(() => {
    navRef.current?.querySelector<HTMLElement>("a[aria-current=page]")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [pathname, topics]);

  return (
    <nav ref={navRef} className="-mb-px mt-3 flex gap-1 overflow-x-auto">
      {topics.map((t, i) => (
        <NavLink
          key={t.topic_id}
          to={`/topic/${t.topic_id}`}
          title={t.title}
          className={({ isActive }) =>
            `max-w-[12rem] shrink-0 truncate border-b-2 px-3 py-2 text-sm transition-colors ${
              isActive
                ? "border-indigo-600 font-medium text-indigo-700"
                : "border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-800"
            }`
          }
        >
          <span className="mr-1.5 text-xs text-slate-400">{i + 1}</span>
          {t.title}
        </NavLink>
      ))}
    </nav>
  );
}
