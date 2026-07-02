// Next.js Route Handler — proxies the browser request to the local FastAPI.
// Keeps the backend URL server-side and lets you swap hosts without touching the UI.
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const body = await req.json();
  const backend = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const r = await fetch(`${backend}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    return NextResponse.json(data, { status: r.status });
  } catch (e: any) {
    return NextResponse.json(
      { error: `Backend unreachable at ${backend}: ${e.message}` },
      { status: 502 }
    );
  }
}
