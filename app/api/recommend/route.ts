
// -------------------------------------------------------------
// File: app/api/recommend/route.ts
import { NextResponse } from "next/server";

// Point this to your Python recommender (Gradio/Flask/FastAPI) if available.
const BACKEND_URL = process.env.RECOMMENDER_URL; // e.g. "http://localhost:8000/recommend"

export async function POST(req: Request) {
  const body = await req.json();
  const { query, genre, year } = body ?? {};

  // If a backend URL is provided, forward the request.
  if (BACKEND_URL) {
    try {
      const r = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, genre, year }),
      });
      const data = await r.json();
      return NextResponse.json(data, { status: r.status });
    } catch (e) {
      return NextResponse.json(
        { reply: "Backend unreachable.", movies: [] },
        { status: 502 }
      );
    }
  }

  // Fallback mock to make the UI work out of the box.
  const mock = [
    {
      id: "1",
      title: "Blade Runner 2049",
      year: 2017,
      genres: ["Sci-Fi", "Thriller"],
      poster: "https://image.tmdb.org/t/p/w342/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
      overview:
        "A young blade runner discovers a long-buried secret that leads him to track down former blade runner Rick Deckard.",
      score: 0.92,
    },
    {
      id: "2",
      title: "Arrival",
      year: 2016,
      genres: ["Sci-Fi", "Drama"],
      poster: "https://image.tmdb.org/t/p/w342/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg",
      overview:
        "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world.",
      score: 0.88,
    },
  ];

  return NextResponse.json({
    reply: `Here are ${genre ? genre + " " : ""}picks${year ? " around " + year : ""} for \"${
      query || "your request"
    }\"`,
    movies: mock,
  });
}
