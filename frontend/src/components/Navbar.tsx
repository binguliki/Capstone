import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useLanguage } from "@/i18n/LanguageContext";
import { useAuth } from "@/context/AuthContext";
import type { Language } from "@/i18n/translations";
import { Home, Search, FileText, Newspaper, Menu, X, Globe, Leaf, ChevronDown, LogIn, LogOut, User } from "lucide-react";
import { Button } from "@/components/ui/button";

const Navbar: React.FC = () => {
  const { language, setLanguage, t } = useLanguage();
  const { isAuthenticated, user, login, logout } = useAuth();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [langOpen, setLangOpen] = useState(false);
  const [userOpen, setUserOpen] = useState(false);

  const links = [
    { to: "/", label: t.nav.home, icon: Home },
    { to: "/detect", label: t.nav.detect, icon: Search },
    { to: "/report", label: t.nav.report, icon: FileText },
    { to: "/news", label: t.nav.news, icon: Newspaper },
  ];

  const languages: { code: Language; label: string }[] = [
    { code: "en", label: t.language.en },
    { code: "hi", label: t.language.hi },
    { code: "te", label: t.language.te },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-card/95 backdrop-blur-md border-b border-border shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 flex-shrink-0">
            <div className="w-8 h-8 gradient-primary rounded-lg flex items-center justify-center">
              <Leaf className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="text-base font-bold text-foreground hidden sm:block">{t.appName}</span>
          </Link>

          {/* Desktop Nav - compact pills */}
          <div className="hidden md:flex items-center gap-0.5 bg-muted/50 rounded-lg p-0.5">
            {links.map((link) => {
              if ((link.to === '/detect' || link.to === '/report') && !isAuthenticated) return null;
              
              const Icon = link.icon;
              const isActive = location.pathname === link.to;
              return (
                <Link
                  key={link.to}
                  to={link.to}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    isActive
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-background"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Language Dropdown */}
          <div className="flex items-center gap-1">
            <div className="relative">
              <button
                onClick={() => setLangOpen(!langOpen)}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <Globe className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">{languages.find(l => l.code === language)?.label}</span>
                <ChevronDown className={`w-3 h-3 transition-transform ${langOpen ? "rotate-180" : ""}`} />
              </button>
              {langOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setLangOpen(false)} />
                  <div className="absolute right-0 top-full mt-1 bg-card border border-border rounded-lg shadow-lg py-1 min-w-[110px] z-50">
                    {languages.map((lang) => (
                      <button
                        key={lang.code}
                        onClick={() => { setLanguage(lang.code); setLangOpen(false); }}
                        className={`w-full text-left px-3 py-1.5 text-xs transition-colors ${
                          language === lang.code
                            ? "bg-primary/10 text-primary font-medium"
                            : "text-foreground hover:bg-muted"
                        }`}
                      >
                        {lang.label}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* Auth Buttons */}
            {isAuthenticated ? (
              <div className="relative ml-2">
                <button
                  onClick={() => setUserOpen(!userOpen)}
                  className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 overflow-hidden border border-border hover:ring-2 ring-primary/50 transition-all"
                >
                  {user?.email ? (
                    <img 
                      src={`https://ui-avatars.com/api/?name=${encodeURIComponent(user.email)}&background=random`} 
                      alt={user?.name || "User"} 
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <User className="w-4 h-4 text-primary" />
                  )}
                </button>
                {userOpen && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setUserOpen(false)} />
                    <div className="absolute right-0 top-full mt-2 bg-card border border-border rounded-lg shadow-lg py-2 min-w-[150px] z-50">
                      <div className="px-4 py-2 border-b border-border/50 mb-1">
                        <p className="text-sm font-medium text-foreground truncate">{user?.name}</p>
                        <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
                      </div>
                      <button
                        onClick={() => {
                          setUserOpen(false);
                          logout();
                        }}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-destructive hover:bg-destructive/10 transition-colors"
                      >
                        <LogOut className="w-4 h-4" />
                        Logout
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <Button
                onClick={login}
                size="sm"
                className="gap-1"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Login</span>
              </Button>
            )}

            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted"
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Nav */}
        {mobileOpen && (
          <div className="md:hidden pb-3 border-t border-border pt-2 space-y-0.5">
            {links.map((link) => {
              if ((link.to === '/detect' || link.to === '/report') && !isAuthenticated) return null;

              const Icon = link.icon;
              const isActive = location.pathname === link.to;
              return (
                <Link
                  key={link.to}
                  to={link.to}
                  onClick={() => setMobileOpen(false)}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {link.label}
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
