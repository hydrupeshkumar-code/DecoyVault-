/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "hero-zoom": "heroZoom 2.5s cubic-bezier(0.16,1,0.3,1) forwards",
        "hero-reveal": "heroReveal 1.2s cubic-bezier(0.16,1,0.3,1) both",
        "hero-fade": "heroFadeUp 1s cubic-bezier(0.16,1,0.3,1) both",
      },
      keyframes: {
        heroZoom: {
          from: { transform: "scale(1.08)" },
          to: { transform: "scale(1)" },
        },
        heroReveal: {
          from: { opacity: "0", clipPath: "inset(0 100% 0 0)" },
          to: { opacity: "1", clipPath: "inset(0 0% 0 0)" },
        },
        heroFadeUp: {
          from: { opacity: "0", transform: "translateY(20px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
