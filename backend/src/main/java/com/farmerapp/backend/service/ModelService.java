package com.farmerapp.backend.service;

import com.farmerapp.backend.entity.ModelCall;
import com.farmerapp.backend.entity.User;
import com.farmerapp.backend.repository.ModelCallRepository;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.UUID;

@Service
public class ModelService {
    private static final Logger logger = LoggerFactory.getLogger(ModelService.class);

    @Autowired
    private ModelCallRepository modelCallRepository;

    @Autowired
    private RestTemplate restTemplate;

    @Value("${model.backend.url}")
    private String modelBackendUrl;

    @Value("${app.upload.dir}")
    private String uploadDir;

    public ModelCall predictDisease(MultipartFile image, User user) throws IOException {
        String imageName = UUID.randomUUID().toString() + "_" + image.getOriginalFilename();
        Path imagePath = Paths.get(uploadDir, imageName);
        Files.createDirectories(imagePath.getParent());
        Files.write(imagePath, image.getBytes());
        String imageUrl = imagePath.toString();

        // Call model backend
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", image.getResource());

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            logger.info("Calling model backend at: {}", modelBackendUrl);
            ResponseEntity<String> response = restTemplate.postForEntity(modelBackendUrl, requestEntity, String.class);

            if (!response.getStatusCode().is2xxSuccessful()) {
                logger.error("Model backend returned status: {}, body: {}", response.getStatusCode(),
                        response.getBody());
                throw new RuntimeException("Model backend error: " + response.getStatusCode());
            }

            JsonObject jsonResponse = JsonParser.parseString(response.getBody()).getAsJsonObject();

            ModelCall modelCall = new ModelCall();
            modelCall.setUser(user);
            modelCall.setImageUrl(imageUrl);
            modelCall.setPredictionLabel(jsonResponse.get("classification").getAsString());
            modelCall.setConfidenceScore(jsonResponse.get("confidence").getAsDouble());
            modelCall.setResponseJson(response.getBody());
            modelCall.setStatus(ModelCall.Status.SUCCESS);
            modelCall.setCreatedAt(LocalDateTime.now());

            logger.info("Successfully predicted disease: {}", modelCall.getPredictionLabel());
            return modelCallRepository.save(modelCall);
        } catch (Exception e) {
            logger.error("Error calling model backend or processing response: {}", e.getMessage(), e);
            ModelCall modelCall = new ModelCall();
            modelCall.setUser(user);
            modelCall.setImageUrl(imageUrl);
            modelCall.setStatus(ModelCall.Status.FAILED);
            modelCall.setCreatedAt(LocalDateTime.now());
            return modelCallRepository.save(modelCall);
        }
    }
}
