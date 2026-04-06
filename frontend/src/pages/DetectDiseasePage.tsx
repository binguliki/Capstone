import React, { useState, useRef, useCallback, useEffect } from "react";
import { useLanguage } from "@/i18n/LanguageContext";
import { Upload, Camera, RotateCcw, AlertTriangle, CheckCircle, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, PredictionResponse, BoundingBox } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface DetectionResult {
  disease: string;
  confidence: number;
  severity: "low" | "medium" | "high";
  boundingBoxes: BoundingBox[];
  advisory: {
    causes: string[];
    prevention: string[];
    remedies: string[];
  };
}

const DetectDiseasePage: React.FC = () => {
  const { t } = useLanguage();
  const { toast } = useToast();
  const [image, setImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loadingAdvisory, setLoadingAdvisory] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
      setSelectedFile(file);
      setResult(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const analyzeImage = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);
    try {
      const response: PredictionResponse = await api.predictDisease(selectedFile);
      const detectionResult: DetectionResult = {
        disease: response.prediction,
        confidence: Number((response.confidence * 100).toFixed(2)),
        severity: (response.confidence * 100) > 80 ? "high" : (response.confidence * 100) > 60 ? "medium" : "low",
        boundingBoxes: response.bounding_boxes || [],
        advisory: {
          causes: [],
          prevention: [],
          remedies: [],
        },
      };
      setResult(detectionResult);
    } catch (error) {
      console.error("Error analyzing image:", error);
      toast({
        title: "❌ Analysis Failed",
        description: "Could not analyze the image. Please try again or check your connection.",
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const fetchAdvisory = async () => {
    if (!result) return;
    setLoadingAdvisory(true);
    try {
      const data = await api.getAdvisory(result.disease);
      setResult({
        ...result,
        advisory: {
          causes: data.causes || [],
          prevention: data.prevention || [],
          remedies: data.remedies || []
        }
      });
    } catch (error) {
      toast({
        title: "Failed to fetch advisory",
        description: "Could not reach the AI service. Please try again.",
      });
    } finally {
      setLoadingAdvisory(false);
    }
  };

  const resetAll = () => {
    setImage(null);
    setSelectedFile(null);
    setResult(null);
    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // Draw bounding boxes on canvas whenever result changes
  useEffect(() => {
    if (!result || !image || result.boundingBoxes.length === 0) return;
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const drawBoxes = () => {
      canvas.width = img.naturalWidth || img.offsetWidth;
      canvas.height = img.naturalHeight || img.offsetHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const W = canvas.width;
      const H = canvas.height;

      result.boundingBoxes.forEach((box) => {
        const x = box.x1 * W;
        const y = box.y1 * H;
        const w = (box.x2 - box.x1) * W;
        const h = (box.y2 - box.y1) * H;

        // Draw box
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = Math.max(2, W / 200);
        ctx.strokeRect(x, y, w, h);

        // Draw label background
        const label = `${box.label} ${(box.confidence * 100).toFixed(1)}%`;
        ctx.font = `bold ${Math.max(12, W / 60)}px Inter, sans-serif`;
        const textWidth = ctx.measureText(label).width;
        const textHeight = Math.max(14, W / 50);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.85)';
        ctx.fillRect(x, Math.max(0, y - textHeight - 4), textWidth + 8, textHeight + 4);

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x + 4, Math.max(textHeight, y - 4));
      });
    };

    if (img.complete && img.naturalWidth > 0) {
      drawBoxes();
    } else {
      img.onload = drawBoxes;
    }
  }, [result, image]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high":
        return "text-destructive";
      case "medium":
        return "text-warning";
      case "low":
        return "text-success";
      default:
        return "text-muted-foreground";
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="text-center mb-10">
        <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">{t.detect.title}</h1>
        <p className="text-muted-foreground text-lg">{t.detect.subtitle}</p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
              dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
            } ${image ? "hidden" : "block"}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files[0];
              if (file) handleFile(file);
            }}
          >
            <Upload className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
            <p className="text-foreground font-medium mb-2">{t.detect.dragDrop}</p>
            <p className="text-sm text-muted-foreground mb-6">PNG, JPG up to 5MB</p>
            
            <div className="flex justify-center gap-4">
              <Button onClick={() => fileInputRef.current?.click()} variant="default">
                <Upload className="w-4 h-4 mr-2" />
                {t.detect.upload}
              </Button>
              <Button onClick={() => cameraInputRef.current?.click()} variant="outline">
                <Camera className="w-4 h-4 mr-2" />
                {t.detect.capture}
              </Button>
            </div>
            
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            />
            <input
              type="file"
              ref={cameraInputRef}
              className="hidden"
              accept="image/*"
              capture="environment"
              onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            />
          </div>

          {/* Image Preview with canvas annotation overlay */}
          {image && (
            <div className="rounded-xl overflow-hidden glass-card relative group animate-fade-in">
              {/* Base image */}
              <img
                ref={imgRef}
                src={image}
                alt="Crop preview"
                className="w-full h-64 object-cover block"
              />
              {/* Canvas overlay - draws bounding boxes on top */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ objectFit: 'cover' }}
              />
              <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <Button onClick={resetAll} variant="secondary" className="gap-2">
                  <RotateCcw className="w-4 h-4" />
                  {t.detect.tryAnother}
                </Button>
              </div>
            </div>
          )}

          {image && !result && (
            <Button
              className="w-full text-lg h-14"
              onClick={analyzeImage}
              disabled={analyzing}
            >
              {analyzing ? (
                <span className="flex items-center gap-2">
                  <div className="w-5 h-5 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                  {t.detect.analyzing}
                </span>
              ) : (
                "Identify Disease"
              )}
            </Button>
          )}
        </div>

        {/* Results Section */}
        <div className="space-y-6">
          {result ? (
            <div className="glass-card p-6 rounded-xl animate-fade-in space-y-6">
              <h3 className="text-2xl font-bold text-foreground border-b border-border pb-4">
                {t.detect.result}
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-background/50 p-4 rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">{t.detect.disease}</p>
                  <p className={`font-semibold text-lg ${result.disease.toLowerCase() === "healthy" ? "text-success" : "text-destructive"}`}>
                    {result.disease}
                  </p>
                </div>
                <div className="bg-background/50 p-4 rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">{t.detect.confidence}</p>
                  <p className="font-semibold text-lg text-foreground">{result.confidence}%</p>
                </div>
              </div>

              {result.disease.toLowerCase() !== "healthy" ? (
                <>
                  <div className="flex items-center gap-2 p-3 bg-warning/10 rounded-lg">
                    <AlertTriangle className={`w-5 h-5 ${getSeverityColor(result.severity)}`} />
                    <span className="text-foreground">
                      {t.detect.severity}: <strong className={getSeverityColor(result.severity)}>
                        {t.detect[result.severity as "low" | "medium" | "high"]}
                      </strong>
                    </span>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold text-primary flex items-center gap-2">
                        <Info className="w-5 h-5" />
                        {t.detect.advisory}
                      </h4>
                      {result.advisory.causes.length === 0 && (
                        <Button 
                          size="sm" 
                          variant="outline" 
                          onClick={fetchAdvisory} 
                          disabled={loadingAdvisory}
                        >
                          {loadingAdvisory ? "Generating..." : "Get Detailed AI Advisory"}
                        </Button>
                      )}
                    </div>
                    
                    {result.advisory.causes.length > 0 && (
                      <>
                        {/* Causes */}
                        <div className="space-y-2">
                          <p className="text-sm font-medium text-foreground">{t.detect.causes}</p>
                          <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                            {result.advisory.causes.map((i, idx) => <li key={idx}>{i}</li>)}
                          </ul>
                        </div>

                        {/* Prevention */}
                        <div className="space-y-2">
                          <p className="text-sm font-medium text-foreground">{t.detect.prevention}</p>
                          <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                            {result.advisory.prevention.map((i, idx) => <li key={idx}>{i}</li>)}
                          </ul>
                        </div>

                        {/* Remedies */}
                        <div className="space-y-2 bg-primary/5 p-4 rounded-lg border border-primary/20">
                          <p className="text-sm font-medium text-primary">{t.detect.remedies}</p>
                          <ul className="list-disc list-inside text-sm text-foreground space-y-1">
                            {result.advisory.remedies.map((i, idx) => <li key={idx}>{i}</li>)}
                          </ul>
                        </div>
                      </>
                    )}
                  </div>
                </>
              ) : (
                <div className="flex items-center gap-3 p-4 bg-success/10 rounded-lg">
                  <CheckCircle className="w-8 h-8 text-success flex-shrink-0" />
                  <p className="text-success font-medium">
                    {t.detect.noDisease}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full min-h-[400px] border-2 border-dashed border-border rounded-xl flex items-center justify-center p-8 text-center text-muted-foreground">
              <p>Upload an image to see the detection results here.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DetectDiseasePage;
