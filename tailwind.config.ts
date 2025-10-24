import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./pages/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [require("@tailwindcss/typography")],
  // Remove @tailwindcss/line-clamp - it's built-in now!
};

export default config;