export type NewsFeedItem = {
  id: string;
  title: string;
  summary: string;
  link: string;
  publishedAt: string;
};

/** India-edition Google News RSS — agriculture-focused queries. */
const INDIA_AGRICULTURE_FEED_URLS = [
  "https://news.google.com/rss/search?q=agriculture+farming+crops+Kharif+Rabi+monsoon+India&hl=en-IN&gl=IN&ceid=IN:en",
  "https://news.google.com/rss/search?q=PM-Kisan+MSP+mandi+farmers+Kisan+agriculture+India&hl=en-IN&gl=IN&ceid=IN:en",
  "https://news.google.com/rss/search?q=organic+fertilizer+irrigation+soil+pesticide+agriculture+India&hl=en-IN&gl=IN&ceid=IN:en",
] as const;

function stripHtml(html: string): string {
  if (typeof document === "undefined") {
    return html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  }
  const d = document.createElement("div");
  d.innerHTML = html;
  return (d.textContent || d.innerText || "").replace(/\s+/g, " ").trim();
}

function parseGoogleNewsRssXml(xml: string): NewsFeedItem[] {
  if (!/<rss[\s>]/i.test(xml)) return [];
  const doc = new DOMParser().parseFromString(xml, "text/xml");
  if (doc.querySelector("parsererror")) return [];

  const channel = doc.querySelector("channel");
  if (!channel) return [];

  const out: NewsFeedItem[] = [];
  channel.querySelectorAll(":scope > item").forEach((el, i) => {
    const title = el.querySelector("title")?.textContent?.trim() || "Untitled";
    const linkEl = el.querySelector("link");
    const link = (linkEl?.textContent ?? linkEl?.getAttribute("href") ?? "").trim();
    const pubDate = el.querySelector("pubDate")?.textContent?.trim() || new Date().toUTCString();
    const descRaw = el.querySelector("description")?.textContent ?? "";
    const guid = el.querySelector("guid")?.textContent?.trim() || link || `ag-${i}`;
    const summaryRaw = stripHtml(descRaw);
    const summary = summaryRaw.length > 280 ? `${summaryRaw.slice(0, 277)}…` : summaryRaw;

    out.push({
      id: guid,
      title,
      summary: summary || "—",
      link,
      publishedAt: pubDate,
    });
  });

  return out.slice(0, 15);
}

/** Dev / `vite preview`: same-origin proxy in vite.config.ts */
async function fetchRssThroughAppProxy(rssUrl: string): Promise<string | null> {
  try {
    const res = await fetch(`/api/fetch-rss?url=${encodeURIComponent(rssUrl)}`);
    if (!res.ok) return null;
    const text = await res.text();
    if (!/<rss[\s>]/i.test(text)) return null;
    return text;
  } catch {
    return null;
  }
}

type GNewsArticle = {
  title: string;
  description: string;
  url: string;
  publishedAt: string;
};

type GNewsResponse = { articles?: GNewsArticle[] };

async function fetchFromGNews(): Promise<NewsFeedItem[]> {
  const key = import.meta.env.VITE_GNEWS_API_KEY?.trim();
  if (!key) return [];

  const buildUrl = (q: string) => {
    const u = new URL("https://gnews.io/api/v4/search");
    u.searchParams.set("q", q);
    u.searchParams.set("lang", "en");
    u.searchParams.set("country", "in");
    u.searchParams.set("max", "10");
    u.searchParams.set("sortby", "publishedAt");
    u.searchParams.set("apikey", key);
    return u.toString();
  };

  const queries = [
    "agriculture farming crops harvest India",
    "farmers Kisan MSP mandi agriculture India",
  ];

  const all: NewsFeedItem[] = [];
  for (const q of queries) {
    try {
      const res = await fetch(buildUrl(q));
      if (!res.ok) continue;
      const data = (await res.json()) as GNewsResponse;
      for (const a of data.articles ?? []) {
        const summaryRaw = stripHtml(a.description || "");
        const summary = summaryRaw.length > 280 ? `${summaryRaw.slice(0, 277)}…` : summaryRaw;
        const id = a.url || a.title;
        all.push({
          id,
          title: (a.title || "Untitled").trim(),
          summary: summary || "—",
          link: a.url,
          publishedAt: a.publishedAt || new Date().toUTCString(),
        });
      }
    } catch {
      /* try next query */
    }
  }
  return all;
}

type Rss2JsonItem = {
  title?: string;
  description?: string;
  content?: string;
  link?: string;
  pubDate?: string;
  guid?: string;
};

type Rss2JsonResponse = {
  status: string;
  items?: Rss2JsonItem[];
  message?: string;
};

function rss2JsonUrl(rssUrl: string): string {
  const apiKey = import.meta.env.VITE_RSS2JSON_API_KEY;
  const base = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssUrl)}`;
  return apiKey ? `${base}&api_key=${encodeURIComponent(apiKey)}` : base;
}

async function fetchOneFeedRss2Json(rssUrl: string): Promise<NewsFeedItem[]> {
  try {
    const res = await fetch(rss2JsonUrl(rssUrl));
    if (!res.ok) return [];
    const data = (await res.json()) as Rss2JsonResponse;
    if (data.status !== "ok" || !Array.isArray(data.items) || data.items.length === 0) {
      return [];
    }

    return data.items.slice(0, 12).map((item, i) => {
      const link = item.link || "";
      const id = String(item.guid || link || `ag-in-${i}`);
      const summaryRaw = stripHtml(item.description || item.content || item.title || "");
      const summary = summaryRaw.length > 280 ? `${summaryRaw.slice(0, 277)}…` : summaryRaw;

      return {
        id,
        title: (item.title || "Untitled").trim(),
        summary: summary || "—",
        link,
        publishedAt: item.pubDate || new Date().toUTCString(),
      };
    });
  } catch {
    return [];
  }
}

function mergeDedupeSort(items: NewsFeedItem[]): NewsFeedItem[] {
  const seen = new Set<string>();
  const unique: NewsFeedItem[] = [];
  for (const it of items) {
    const key = it.link || it.id;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(it);
  }
  unique.sort((a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime());
  return unique.slice(0, 24);
}

export async function fetchAggregatedNews(): Promise<NewsFeedItem[]> {
  const fromProxy: NewsFeedItem[] = [];
  for (const rssUrl of INDIA_AGRICULTURE_FEED_URLS) {
    const xml = await fetchRssThroughAppProxy(rssUrl);
    if (xml) fromProxy.push(...parseGoogleNewsRssXml(xml));
  }
  if (fromProxy.length) return mergeDedupeSort(fromProxy);

  const fromGNews = await fetchFromGNews();
  if (fromGNews.length) return mergeDedupeSort(fromGNews);

  const fromRss2: NewsFeedItem[] = [];
  for (const rssUrl of INDIA_AGRICULTURE_FEED_URLS) {
    fromRss2.push(...(await fetchOneFeedRss2Json(rssUrl)));
  }
  if (fromRss2.length) return mergeDedupeSort(fromRss2);

  throw new Error(
    "No agriculture news sources responded. Use npm run dev or vite preview (built-in RSS proxy), or add a free key: VITE_GNEWS_API_KEY from https://gnews.io or VITE_RSS2JSON_API_KEY from https://rss2json.com — then rebuild.",
  );
}
