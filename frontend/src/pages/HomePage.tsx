import React from "react";
import { Link } from "react-router-dom";
import { useLanguage } from "@/i18n/LanguageContext";
import { Search, FileText, Newspaper } from "lucide-react";
import farmerMascot from "@/assets/farmer-mascot.png";
import detectImg from "@/assets/detect-disease.png";
import reportImg from "@/assets/report-disease.png";
import newsImg from "@/assets/news-icon.png";
import heroBg from "@/assets/hero-bg.jpg";

const HomePage: React.FC = () => {
  const { t } = useLanguage();

  const cards = [
    { to: "/detect", title: t.home.detectCard, desc: t.home.detectDesc, icon: Search, img: detectImg, color: "from-primary to-success" },
    { to: "/report", title: t.home.reportCard, desc: t.home.reportDesc, icon: FileText, img: reportImg, color: "from-warning to-accent" },
    { to: "/news", title: t.home.newsCard, desc: t.home.newsDesc, icon: Newspaper, img: newsImg, color: "from-primary to-primary" },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative overflow-hidden gradient-hero">
        <div
          className="absolute inset-0 opacity-20 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroBg})` }}
        />
        <div className="relative container mx-auto px-4 py-12 md:py-20">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-1 text-center md:text-left">
              <h1 className="text-3xl md:text-5xl font-bold text-primary-foreground mb-4 animate-fade-in">
                {t.home.welcome}
              </h1>
              <p className="text-lg md:text-xl text-primary-foreground/80 max-w-lg">
                {t.home.subtitle}
              </p>
            </div>
            <div className="flex-shrink-0">
              <img
                src={farmerMascot}
                alt="Farmer mascot"
                width={200}
                height={200}
                className="drop-shadow-2xl animate-fade-in"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="container mx-auto px-4 py-10 md:py-16">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-4xl mx-auto">
          {cards.map((card, i) => {
            const Icon = card.icon;
            return (
              <Link
                key={card.to}
                to={card.to}
                className="group card-hover glass-card p-6 flex flex-col items-center text-center gap-4"
                style={{ animationDelay: `${i * 100}ms` }}
              >
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br opacity-90 flex items-center justify-center overflow-hidden">
                  <img
                    src={card.img}
                    alt=""
                    width={64}
                    height={64}
                    loading="lazy"
                    className="w-16 h-16 object-contain"
                  />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-foreground mb-2 group-hover:text-primary transition-colors">
                    {card.title}
                  </h2>
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {card.desc}
                  </p>
                </div>
                <div className="mt-auto pt-2">
                  <span className="inline-flex items-center gap-1 text-sm font-medium text-primary">
                    <Icon className="w-4 h-4" />
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default HomePage;
