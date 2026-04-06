import React from "react";
import { createContext, useContext, useState } from "react";

export interface DiseaseReport {
  id: string;
  image: string;
  cropType: string;
  location: string;
  description?: string;
  date: string;
  points: number;
}

interface AppState {
  reports: DiseaseReport[];
  totalPoints: number;
  addReport: (report: Omit<DiseaseReport, "id" | "date" | "points">) => void;
}

const AppContext = createContext<AppState | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [reports, setReports] = useState<DiseaseReport[]>(() => {
    const saved = localStorage.getItem("krishi-reports");
    return saved ? JSON.parse(saved) : [];
  });

  const totalPoints = reports.reduce((sum, r) => sum + r.points, 0);

  const addReport = (report: Omit<DiseaseReport, "id" | "date" | "points">) => {
    const newReport: DiseaseReport = {
      ...report,
      id: crypto.randomUUID(),
      date: new Date().toLocaleDateString(),
      points: 10,
    };
    const updated = [newReport, ...reports];
    setReports(updated);
    localStorage.setItem("krishi-reports", JSON.stringify(updated));
  };

  return (
    <AppContext.Provider value={{ reports, totalPoints, addReport }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppState = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppState must be used within AppProvider");
  return ctx;
};
