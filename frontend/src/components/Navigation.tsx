import { useState } from "react";

const NAV_ITEMS = ["Demo", "Architecture", "Metrics", "Datasets", "Results"] as const;
type NavItem = (typeof NAV_ITEMS)[number];

export default function Navigation() {
  const [active, setActive] = useState<NavItem>("Demo");

  return (
    <nav
      aria-label="Main navigation"
      className="fixed top-0 left-0 right-0 z-[100] flex items-center justify-between p-4 sm:p-5"
    >
      {/* ── Left: Wordmark ── */}
      <div className="flex items-center gap-2.5">
        <svg
          width="26"
          height="26"
          viewBox="0 0 26 26"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
          focusable="false"
        >
          <rect x="9" y="9" width="8" height="8" rx="1.5" fill="white" />
          <rect x="0" y="11" width="7" height="4" rx="1" fill="white" opacity="0.7" />
          <rect x="19" y="11" width="7" height="4" rx="1" fill="white" opacity="0.7" />
          <line x1="13" y1="0" x2="13" y2="8" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="13" cy="0" r="1.5" fill="#f97316" />
          <path
            d="M 4 22 Q 13 14 22 22"
            stroke="white"
            strokeWidth="1"
            strokeOpacity="0.35"
            fill="none"
            strokeDasharray="2 2"
          />
        </svg>

        <span className="text-white text-2xl font-semibold tracking-tight">
          CloudVision AI
        </span>
      </div>

      {/* ── Center: Glassmorphism pill nav (desktop only) ── */}
      <div className="hidden md:flex absolute left-1/2 -translate-x-1/2">
        <div className="bg-white/20 backdrop-blur-md border border-white/30 rounded-full px-2 py-2 flex items-center gap-1">
          {NAV_ITEMS.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setActive(item)}
              aria-current={active === item ? "page" : undefined}
              className={[
                "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200",
                "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white/60",
                active === item
                  ? "bg-white text-black"
                  : "text-white/80 hover:bg-white/20 hover:text-white",
              ].join(" ")}
            >
              {item}
            </button>
          ))}
        </div>
      </div>

      {/* ── Right: Launch Demo (desktop only) ── */}
      <div className="hidden md:block">
        <button
          type="button"
          className="bg-white text-gray-900 text-sm font-semibold px-6 py-2.5 rounded-full hover:bg-gray-100 transition-all duration-200 hover:scale-[1.02] active:scale-95 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white/60"
        >
          Launch Demo
        </button>
      </div>
    </nav>
  );
}
