const API_BASE_URL = 'http://localhost:8080';

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  confidence: number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  status: string;
  remedies?: string;
  bounding_boxes?: BoundingBox[];
}

export const api = {
  predictDisease: async (imageFile: File): Promise<PredictionResponse> => {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/api/model/predict`, {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to predict disease');
    }

    return response.json();
  },

  getAdvisory: async (disease: string): Promise<{ causes: string[]; prevention: string[]; remedies: string[] }> => {
    const response = await fetch(
      `${API_BASE_URL}/api/model/advise?disease=${encodeURIComponent(disease)}`,
      { credentials: 'include' }
    );
    if (!response.ok) throw new Error('Failed to fetch advisory');
    return response.json();
  },

  createReport: async (imageFiles: File[], description: string) => {
    const formData = new FormData();
    imageFiles.forEach((file, index) => {
      formData.append('images', file);
    });
    formData.append('description', description);

    const response = await fetch(`${API_BASE_URL}/api/reports`, {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to create report');
    }

    return response.json();
  },

  getUserReports: async () => {
    const response = await fetch(`${API_BASE_URL}/api/reports/my`, {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to fetch reports');
    }

    return response.json();
  },

  getUser: async () => {
    const response = await fetch(`${API_BASE_URL}/user`, {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user');
    }

    return response.json();
  },

  login: () => {
    window.location.href = `${API_BASE_URL}/oauth2/authorization/google`;
  },

  logout: async () => {
    // For OAuth, perhaps redirect to logout
    window.location.href = `${API_BASE_URL}/logout`;
  },
};