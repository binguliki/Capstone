package com.farmerapp.backend.service;

import com.farmerapp.backend.entity.DiseaseReport;
import com.farmerapp.backend.entity.User;
import com.farmerapp.backend.repository.DiseaseReportRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
public class DiseaseReportService {

    @Autowired
    private DiseaseReportRepository diseaseReportRepository;

    @Value("${app.upload.dir}")
    private String uploadDir;

    public DiseaseReport createReport(List<MultipartFile> images, String description, User user) throws IOException {
        List<String> imageUrls = new ArrayList<>();
        for (MultipartFile image : images) {
            String imageName = UUID.randomUUID().toString() + "_" + image.getOriginalFilename();
            Path imagePath = Paths.get(uploadDir, imageName);
            Files.createDirectories(imagePath.getParent());
            Files.write(imagePath, image.getBytes());
            imageUrls.add(imagePath.toString());
        }

        DiseaseReport report = new DiseaseReport();
        report.setUser(user);
        report.setImageUrls(imageUrls);
        report.setDescription(description);
        report.setStatus(DiseaseReport.Status.PENDING);
        report.setCreatedAt(LocalDateTime.now());
        report.setUpdatedAt(LocalDateTime.now());

        return diseaseReportRepository.save(report);
    }

    public List<DiseaseReport> getReportsByUser(Long userId) {
        return diseaseReportRepository.findByUser_UserId(userId);
    }

    public List<DiseaseReport> getAllReports() {
        return diseaseReportRepository.findAll();
    }

    public DiseaseReport updateReportStatus(Long reportId, DiseaseReport.Status status, String adminNotes) {
        DiseaseReport report = diseaseReportRepository.findById(reportId).orElseThrow();
        report.setStatus(status);
        report.setAdminNotes(adminNotes);
        report.setUpdatedAt(LocalDateTime.now());
        return diseaseReportRepository.save(report);
    }
}