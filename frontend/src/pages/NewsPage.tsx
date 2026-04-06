import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLanguage } from "@/i18n/LanguageContext";
import type { Language } from "@/i18n/translations";
import { ExternalLink, RefreshCw } from "lucide-react";
import { fetchAggregatedNews, type NewsFeedItem } from "@/lib/newsFeed";
import { translateText, type TranslateTarget } from "@/lib/translateText";
import { Button } from "@/components/ui/button";

function formatPublished(iso: string, locale: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(locale, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function localeForLang(lang: Language): string {
  if (lang === "hi") return "hi-IN";
  if (lang === "te") return "te-IN";
  return "en-IN";
}

export default function NewsPage() {
  const { language, t } = useLanguage();
  const [translated, setTranslated] = useState<Record<string, { title: string; summary: string }>>({});
  const [translating, setTranslating] = useState(false);

  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } = useQuery({
    queryKey: ["ag-news-feed-india"],
    queryFn: fetchAggregatedNews,
    staleTime: 3 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
  });

  const items = useMemo(() => data ?? [], [data]);

  useEffect(() => {
    let cancelled = false;
    if (language === "en" || !items.length) {
      setTranslated({});
      setTranslating(false);
      return;
    }
    const target = language as TranslateTarget;
    setTranslating(true);
    (async () => {
      const next: Record<string, { title: string; summary: string }> = {};
      for (const it of items) {
        if (cancelled) return;
        const [titleT, summaryT] = await Promise.all([
          translateText(it.title, target),
          translateText(it.summary, target),
        ]);
        next[it.id] = { title: titleT, summary: summaryT };
      }
      if (!cancelled) {
        setTranslated(next);
        setTranslating(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [language, items]);

  const displayItems = useMemo(() => {
    return items.map((it: NewsFeedItem) => {
      if (language === "en") return { ...it, title: it.title, summary: it.summary };
      const tr = translated[it.id];
      return {
        ...it,
        title: tr?.title ?? it.title,
        summary: tr?.summary ?? it.summary,
      };
    });
  }, [items, language, translated]);

  const locale = localeForLang(language);

  return (
    <div className="container mx-auto px-4 py-8 max-w-3xl">
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-foreground mb-2">{t.news.title}</h1>
        <p className="text-muted-foreground text-sm mb-4 max-w-md mx-auto">{t.news.subtitle}</p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={isFetching}
            onClick={() => refetch()}
            className="gap-2"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? "animate-spin" : ""}`} />
            {t.news.refresh}
          </Button>
          {dataUpdatedAt ? (
            <span className="text-xs text-muted-foreground">
              {formatPublished(new Date(dataUpdatedAt).toISOString(), locale)}
            </span>
          ) : null}
        </div>
        {translating ? (
          <p className="text-xs text-muted-foreground mt-2">{t.news.translating}</p>
        ) : null}
      </div>

      {isLoading ? (
        <p className="text-center text-muted-foreground py-12">{t.news.loading}</p>
      ) : isError ? (
        <p className="text-center text-destructive py-12">
          {t.news.error}
          {error instanceof Error ? ` (${error.message})` : ""}
        </p>
      ) : !displayItems.length ? (
        <p className="text-center text-muted-foreground py-12">{t.news.empty}</p>
      ) : (
        <ul className="space-y-4 list-none p-0 m-0">
          {displayItems.map((article, i) => (
            <li key={article.id}>
              <article
                className="glass-card p-5 card-hover animate-fade-in border border-border/60 border-l-2 border-l-primary/40"
                style={{ animationDelay: `${Math.min(i, 12) * 45}ms` }}
              >
                <time className="text-xs text-muted-foreground block mb-2" dateTime={article.publishedAt}>
                  {formatPublished(article.publishedAt, locale)}
                </time>
                <h2 className="text-lg font-bold text-foreground leading-snug mb-2">{article.title}</h2>
                <p className="text-sm text-muted-foreground leading-relaxed line-clamp-4 mb-3">
                  {article.summary}
                </p>
                {article.link ? (
                  <a
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline w-fit"
                  >
                    {t.news.readMore}
                    <ExternalLink className="w-3.5 h-3.5 shrink-0" />
                  </a>
                ) : null}
              </article>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
