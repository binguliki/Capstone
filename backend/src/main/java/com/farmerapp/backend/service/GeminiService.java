package com.farmerapp.backend.service;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class GeminiService {
    private static final Logger logger = LoggerFactory.getLogger(GeminiService.class);

    @Value("${gemini.api.key}")
    private String apiKey;

    @Value("${gemini.api.url}")
    private String apiUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    public String getAdvisoryJson(String disease) {
        String prompt = "You are an agricultural expert. The crop has " + disease
                + ". Provide a JSON response with exactly these keys: 'causes' (array of strings), 'prevention' (array of strings), and 'remedies' (array of strings). Do not use markdown blocks, just return raw JSON.";

        JsonObject partObj = new JsonObject();
        partObj.addProperty("text", prompt);

        JsonArray partsArray = new JsonArray();
        partsArray.add(partObj);

        JsonObject contentObj = new JsonObject();
        contentObj.add("parts", partsArray);

        JsonArray contentsArray = new JsonArray();
        contentsArray.add(contentObj);

        JsonObject requestBody = new JsonObject();
        requestBody.add("contents", contentsArray);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<String> entity = new HttpEntity<>(requestBody.toString(), headers);

        String fullUrl = apiUrl + apiKey;

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(fullUrl, entity, String.class);
            if (!response.getStatusCode().is2xxSuccessful()) {
                logger.error("Gemini API returned non-2xx status: {} body={}", response.getStatusCode(),
                        response.getBody());
                throw new IllegalStateException("Non-success response from Gemini API");
            }

            JsonObject jsonResponse = JsonParser.parseString(response.getBody()).getAsJsonObject();
            String responseText = jsonResponse.getAsJsonArray("candidates").get(0).getAsJsonObject()
                    .getAsJsonObject("content").getAsJsonArray("parts").get(0).getAsJsonObject()
                    .get("text").getAsString();
            responseText = responseText.replaceAll("^```json", "").replaceAll("```$", "").trim();
            if (responseText.startsWith("```")) {
                responseText = responseText.substring(responseText.indexOf('\n') + 1);
            }
            return responseText;
        } catch (Exception e) {
            logger.error("Failed to fetch advisory from Gemini", e);
            return "{\"causes\":[], \"prevention\":[], \"remedies\":[\"Unable to fetch data from AI.\"]}";
        }
    }
}
