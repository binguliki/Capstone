/**
 * Translation order:
 * 1) Optional `VITE_TRANSLATION_PROXY_URL` — your backend calling Google Cloud Translation (recommended for production).
 * 2) Lingva (open-source Google Translate frontend) and MyMemory as browser fallbacks.
 */
const MAX_LEN = 450;

export type TranslateTarget = "hi" | "te";

async function translateViaProxy(text: string, target: TranslateTarget): Promise<string | null> {
  const base = import.meta.env.VITE_TRANSLATION_PROXY_URL?.replace(/\/$/, "");
  if (!base) return null;
  const res = await fetch(`${base}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text.slice(0, MAX_LEN), target }),
  });
  if (!res.ok) return null;
  const data = (await res.json()) as { translation?: string };
  return data.translation?.trim() || null;
}

async function translateLingva(text: string, target: TranslateTarget): Promise<string | null> {
  const chunk = text.slice(0, MAX_LEN);
  const url = `https://lingva.ml/api/v1/en/${target}/${encodeURIComponent(chunk)}`;
  const res = await fetch(url);
  if (!res.ok) return null;
  const data = (await res.json()) as { translation?: string };
  return data.translation?.trim() || null;
}

async function translateMyMemory(text: string, target: TranslateTarget): Promise<string | null> {
  const chunk = text.slice(0, MAX_LEN);
  const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(chunk)}&langpair=en|${target}`;
  const res = await fetch(url);
  if (!res.ok) return null;
  const data = (await res.json()) as {
    responseStatus?: number;
    responseData?: { translatedText?: string };
  };
  if (data.responseStatus !== 200 || !data.responseData?.translatedText) return null;
  return data.responseData.translatedText.trim();
}

export async function translateText(text: string, target: TranslateTarget): Promise<string> {
  const t = text.trim();
  if (!t) return text;

  try {
    const proxied = await translateViaProxy(t, target);
    if (proxied) return proxied;
  } catch {
    /* continue */
  }

  try {
    const lingva = await translateLingva(t, target);
    if (lingva) return lingva;
  } catch {
    /* try next */
  }

  try {
    const mm = await translateMyMemory(t, target);
    if (mm) return mm;
  } catch {
    /* fall through */
  }

  return text;
}
