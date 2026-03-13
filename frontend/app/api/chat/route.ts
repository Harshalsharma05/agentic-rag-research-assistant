import { NextResponse } from "next/server";

type ChatRequest = {
  query?: string;
  session_id?: string;
};

function getBackendUrl() {
  // Prefer server-only env var. Fall back to NEXT_PUBLIC for compatibility.
  const baseUrl =
    process.env.BACKEND_URL ||
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    "http://127.0.0.1:8000";

  return baseUrl.replace(/\/$/, "");
}

export async function POST(req: Request) {
  let payload: ChatRequest;

  try {
    payload = (await req.json()) as ChatRequest;
  } catch {
    return NextResponse.json(
      { detail: "Invalid JSON payload." },
      { status: 400 },
    );
  }

  if (!payload?.query || !payload?.session_id) {
    return NextResponse.json(
      { detail: "Both query and session_id are required." },
      { status: 400 },
    );
  }

  const backendUrl = getBackendUrl();

  try {
    const backendRes = await fetch(`${backendUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: payload.query,
        session_id: payload.session_id,
      }),
      cache: "no-store",
    });

    const text = await backendRes.text();
    let body: unknown = null;

    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = { detail: text };
      }
    }

    if (!backendRes.ok) {
      const detail =
        typeof body === "object" && body && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : `Backend returned HTTP ${backendRes.status}`;

      return NextResponse.json({ detail }, { status: backendRes.status });
    }

    return NextResponse.json(body ?? {}, { status: 200 });
  } catch {
    return NextResponse.json(
      {
        detail:
          "Cannot reach backend service. Verify BACKEND_URL (or NEXT_PUBLIC_BACKEND_URL) and backend health.",
      },
      { status: 502 },
    );
  }
}
