package com.farmerapp.backend.controller;

import com.farmerapp.backend.entity.ModelCall;
import com.farmerapp.backend.entity.User;
import com.farmerapp.backend.service.*;
import com.farmerapp.backend.service.UserService;
import org.springframework.http.HttpStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/model")
public class ModelController {

    @Autowired
    private ModelService modelService;

    @Autowired
    private GeminiService geminiService;

    @Autowired
    private UserService userService;

    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predictDisease(
            @RequestParam("image") MultipartFile image,
            @AuthenticationPrincipal OAuth2User principal) throws IOException {

        if (principal == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("error", "Unauthorized"));
        }

        String email = principal.getAttribute("email");
        User user = userService.findByEmail(email).orElseThrow();

        ModelCall modelCall = modelService.predictDisease(image, user);

        if (modelCall.getStatus() == ModelCall.Status.FAILED) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to analyze image. Please check backend logs."));
        }

        Map<String, Object> response = new HashMap<>();
        response.put("prediction", modelCall.getPredictionLabel());
        response.put("confidence", modelCall.getConfidenceScore());
        response.put("status", modelCall.getStatus());

        // Forward bounding_boxes from model-backend response to frontend
        try {
            com.google.gson.JsonObject jsonResponse = com.google.gson.JsonParser
                    .parseString(modelCall.getResponseJson()).getAsJsonObject();
            if (jsonResponse.has("bounding_boxes") && !jsonResponse.get("bounding_boxes").isJsonNull()) {
                Object boundingBoxes = new com.google.gson.Gson().fromJson(
                        jsonResponse.get("bounding_boxes"), Object.class);
                response.put("bounding_boxes", boundingBoxes);
            }
        } catch (Exception e) {
            // Ignore if missing or parsing fails
        }

        return ResponseEntity.ok(response);
    }

    @GetMapping("/advise")
    public ResponseEntity<String> getAdvisory(String disease) {
        String advisoryJson = geminiService.getAdvisoryJson(disease);
        return ResponseEntity.ok()
                .header("Content-Type", "application/json")
                .body(advisoryJson);
    }
}