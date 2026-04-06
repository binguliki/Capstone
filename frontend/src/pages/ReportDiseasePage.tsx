import React, { useState, useRef, useEffect } from "react";
import { useLanguage } from "@/i18n/LanguageContext";
import { useAppState } from "@/context/AppContext";
import { Upload, X, Send, MapPin, Phone } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/lib/api";

interface FormErrors {
  image?: string;
  cropType?: string;
  location?: string;
}

const ReportDiseasePage: React.FC = () => {
  const { t } = useLanguage();
  const { addReport } = useAppState();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mapRef = useRef<HTMLDivElement>(null);

  const [image, setImage] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [cropType, setCropType] = useState("");
  const [location, setLocation] = useState("");
  const [latLng, setLatLng] = useState<{ lat: number; lng: number } | null>(null);
  const [description, setDescription] = useState("");
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Load Leaflet CSS + JS dynamically
  useEffect(() => {
    if (document.getElementById("leaflet-css")) {
      setMapLoaded(true);
      return;
    }
    const link = document.createElement("link");
    link.id = "leaflet-css";
    link.rel = "stylesheet";
    link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
    document.head.appendChild(link);

    const script = document.createElement("script");
    script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
    script.onload = () => setMapLoaded(true);
    document.head.appendChild(script);
  }, []);

  // Initialize map once loaded
  useEffect(() => {
    if (!mapLoaded || !mapRef.current) return;
    // Prevent re-init
    if ((mapRef.current as any)._leaflet_id) return;

    const L = (window as any).L;
    if (!L) return;

    const map = L.map(mapRef.current).setView([20.5937, 78.9629], 5);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap",
    }).addTo(map);

    let marker: any = null;

    map.on("click", async (e: any) => {
      const { lat, lng } = e.latlng;
      setLatLng({ lat, lng });

      if (marker) marker.setLatLng([lat, lng]);
      else marker = L.marker([lat, lng]).addTo(map);

      // Reverse geocode
      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=14`
        );
        const data = await res.json();
        const name = data.display_name || `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
        setLocation(name);
        setErrors((prev) => ({ ...prev, location: undefined }));
      } catch {
        setLocation(`${lat.toFixed(4)}, ${lng.toFixed(4)}`);
      }
    });

    // Try to get user's location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          map.setView([pos.coords.latitude, pos.coords.longitude], 12);
        },
        () => { }
      );
    }

    return () => {
      map.remove();
    };
  }, [mapLoaded]);

  const validateFile = (file: File): string | null => {
    if (!["image/jpeg", "image/png"].includes(file.type)) return t.report.errors.invalidType;
    if (file.size > 5 * 1024 * 1024) return t.report.errors.tooLarge;
    return null;
  };

  const handleFile = (files: FileList) => {
    const fileArray = Array.from(files);
    const validFiles: File[] = [];
    let error: string | null = null;

    for (const file of fileArray) {
      const fileError = validateFile(file);
      if (fileError) {
        error = fileError;
        break;
      }
      validFiles.push(file);
    }

    if (error) {
      setErrors((prev) => ({ ...prev, image: error }));
      return;
    }

    setErrors((prev) => ({ ...prev, image: undefined }));
    setSelectedFiles(validFiles);

    // For preview, show the first image
    if (validFiles.length > 0) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target?.result as string);
      reader.readAsDataURL(validFiles[0]);
    }
  };

  const validate = (): boolean => {
    const newErrors: FormErrors = {};
    if (selectedFiles.length === 0) newErrors.image = t.report.errors.imageRequired;
    if (!cropType.trim()) newErrors.cropType = t.report.errors.cropRequired;
    if (!location.trim()) newErrors.location = t.report.errors.locationRequired;
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate() || selectedFiles.length === 0) return;

    setSubmitting(true);
    try {
      await api.createReport(selectedFiles, description);
      toast({ title: "✅", description: t.report.success });
      setImage(null);
      setSelectedFiles([]);
      setCropType("");
      setLocation("");
      setLatLng(null);
      setDescription("");
    } catch (error) {
      toast({ title: "❌", description: "Failed to submit report" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">{t.report.title}</h1>
        <p className="text-muted-foreground">{t.report.subtitle}</p>
      </div>

      {/* Associate call notice */}
      <div className="mb-6 flex items-start gap-3 rounded-lg border border-primary/20 bg-primary/5 p-4">
        <Phone className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
        <p className="text-sm text-foreground">
          {t.report.callNotice}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="glass-card p-6 space-y-6">
        {/* Image Upload */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">{t.report.imageLabel}</label>
          {image ? (
            <div className="relative">
              <img src={image} alt="Preview" className="w-full max-h-60 object-contain rounded-lg border border-border" />
              <div className="absolute top-2 right-2 bg-primary text-primary-foreground px-2 py-1 rounded text-xs">
                {selectedFiles.length} image{selectedFiles.length > 1 ? 's' : ''}
              </div>
              <button
                type="button"
                onClick={() => { setImage(null); setSelectedFiles([]); }}
                className="absolute top-2 left-2 bg-destructive text-destructive-foreground rounded-full p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:border-primary transition-colors"
            >
              <Upload className="w-10 h-10 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">{t.report.imageRequired} (Multiple images allowed)</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && handleFile(e.target.files)}
          />
          {errors.image && <p className="text-sm text-destructive mt-1">{errors.image}</p>}
        </div>

        {/* Crop Type */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">{t.report.cropType}</label>
          <Input
            value={cropType}
            onChange={(e) => setCropType(e.target.value)}
            placeholder={t.report.cropPlaceholder}
          />
          {errors.cropType && <p className="text-sm text-destructive mt-1">{errors.cropType}</p>}
        </div>

        {/* Location - Map Picker */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            <MapPin className="w-4 h-4 inline mr-1" />
            {t.report.location}
          </label>
          <p className="text-xs text-muted-foreground mb-2">{t.report.mapHint}</p>
          <div
            ref={mapRef}
            className="w-full h-56 rounded-lg border border-border overflow-hidden mb-2"
            style={{ zIndex: 0 }}
          />
          {location && (
            <p className="text-xs text-foreground bg-muted rounded-md px-3 py-2 break-words">
              📍 {location}
            </p>
          )}
          {errors.location && <p className="text-sm text-destructive mt-1">{errors.location}</p>}
        </div>

        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">{t.report.description}</label>
          <Textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder={t.report.descriptionPlaceholder}
            rows={3}
          />
        </div>

        <Button type="submit" className="w-full gap-2 text-lg py-6" disabled={submitting}>
          {submitting ? (
            <span className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
              {t.report.submitting}
            </span>
          ) : (
            <>
              <Send className="w-5 h-5" />
              {t.report.submit}
            </>
          )}
        </Button>
      </form>
    </div>
  );
};

export default ReportDiseasePage;
