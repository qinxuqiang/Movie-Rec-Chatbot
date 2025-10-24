// File: app/page.tsx
"use client";
import { useState, useRef, useEffect } from "react";

type Message = { role: "user" | "assistant"; content: string };

type Movie = {
  id: string;
  title: string;
  year?: number;
  poster?: string | null;
  overview?: string;
  genres?: string[];
  score?: number;
};

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hey! Ask me for a movie and I'll curate picks." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [movies, setMovies] = useState<Movie[]>([]);
  const [genre, setGenre] = useState<string>("");
  const [year, setYear] = useState<string>("");
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages((m) => [...m, userMsg]);
    setLoading(true);
    setInput("");

    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.content, genre, year }),
      });

      if (!res.ok) throw new Error("Request failed");
      const data: { reply: string; movies: Movie[] } = await res.json();

      setMessages((m) => [...m, { role: "assistant", content: data.reply }]);
      setMovies(data.movies || []);
    } catch (err: any) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "Sorry—something went wrong reaching the recommender. Try again in a moment.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8 grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-6">
        {/* Chat + Results */}
        <section className="space-y-6">
          <header className="rounded-2xl p-6 bg-white/5 border border-white/10 backdrop-blur">
            <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">🎬 ReelTalk</h1>
            <p className="text-slate-300 mt-1">Your personal movie recommendation expert.</p>
          </header>

          {/* Chat */}
          <div className="rounded-2xl bg-white/5 border border-white/10 backdrop-blur flex flex-col h-[520px]">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((m, i) => (
                <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 shadow-sm border text-sm md:text-base ${
                      m.role === "user"
                        ? "bg-indigo-600/90 border-indigo-400/30"
                        : "bg-slate-800/80 border-white/10"
                    }`}
                  >
                    {m.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="text-slate-300 text-sm italic animate-pulse">Thinking…</div>
              )}
              <div ref={endRef} />
            </div>

            <form onSubmit={onSubmit} className="p-3 border-t border-white/10">
              <div className="flex gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask for a vibe, genre, or mood…"
                  className="flex-1 bg-slate-900/60 border border-white/10 rounded-xl px-4 py-3 outline-none focus:border-indigo-400"
                />
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-xl px-4 py-3 bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50"
                >
                  Send
                </button>
              </div>
            </form>
          </div>

          {/* Recommendations */}
          <div className="space-y-3">
            {movies.length > 0 && (
              <h2 className="text-lg font-medium text-slate-200">Recommended</h2>
            )}
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {movies.map((m) => (
                <article
                  key={m.id}
                  className="rounded-2xl overflow-hidden bg-white/5 border border-white/10"
                >
                  <div className="aspect-[2/3] bg-slate-800 flex items-center justify-center">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    {m.poster ? (
                      <img src={m.poster} alt={m.title} className="h-full w-full object-cover" />
                    ) : (
                      <div className="text-slate-400 text-sm">No poster</div>
                    )}
                  </div>
                  <div className="p-4 space-y-1">
                    <h3 className="font-semibold leading-tight">{m.title}</h3>
                    <p className="text-xs text-slate-400">
                      {m.year ? m.year : ""} {m.genres?.length ? `• ${m.genres.join(", ")}` : ""}
                    </p>
                    {typeof m.score === "number" && (
                      <p className="text-xs text-slate-400">Match score: {(m.score * 100).toFixed(0)}%</p>
                    )}
                    {m.overview && (
                      <p className="text-sm text-slate-300 line-clamp-3">{m.overview}</p>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* Sidebar */}
        <aside className="lg:sticky top-6 h-fit space-y-4">
          <div className="rounded-2xl p-5 bg-white/5 border border-white/10">
            <h3 className="font-medium mb-3">Filters</h3>
            <div className="space-y-3">
              <div className="space-y-1">
                <label className="text-sm text-slate-300">Genre</label>
                <select
                  value={genre}
                  onChange={(e) => setGenre(e.target.value)}
                  className="w-full bg-slate-900/60 border border-white/10 rounded-xl px-3 py-2"
                >
                  <option value="">Any</option>
                  <option>Sci-Fi</option>
                  <option>Drama</option>
                  <option>Comedy</option>
                  <option>Action</option>
                  <option>Thriller</option>
                  <option>Romance</option>
                  <option>Horror</option>
                  <option>Animation</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-sm text-slate-300">Year</label>
                <input
                  value={year}
                  onChange={(e) => setYear(e.target.value)}
                  inputMode="numeric"
                  placeholder="e.g., 2019"
                  className="w-full bg-slate-900/60 border border-white/10 rounded-xl px-3 py-2"
                />
              </div>
              <button
                onClick={() => {
                  setGenre("");
                  setYear("");
                }}
                className="w-full rounded-xl px-4 py-2 bg-slate-800 hover:bg-slate-700"
              >
                Reset
              </button>
            </div>
          </div>

          <div className="rounded-2xl p-5 bg-white/5 border border-white/10">
            <h3 className="font-medium mb-2">Examples</h3>
            <div className="flex flex-wrap gap-2">
              {["Feel-good sci-fi", "Dark detective noir", "Family weekend"].map((ex) => (
                <button
                  key={ex}
                  onClick={() => setInput(ex)}
                  className="px-3 py-1.5 text-sm rounded-full bg-slate-800 hover:bg-slate-700"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}




// -------------------------------------------------------------
// Quick start:
// 1) npx create-next-app@latest reeltalk --ts --app --eslint --tailwind
// 2) Replace generated files with the ones above (keep app/layout.tsx and import ./globals.css there).
// 3) Run: RECOMMENDER_URL="http://localhost:8000/recommend" npm run dev
// 4) Open http://localhost:3000
