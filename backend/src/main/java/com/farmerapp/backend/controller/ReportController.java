package com.farmerapp.backend.controller;

import com.farmerapp.backend.entity.DiseaseReport;
import com.farmerapp.backend.entity.User;
import com.farmerapp.backend.service.DiseaseReportService;
import com.farmerapp.backend.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/reports")
public class ReportController {

    @Autowired
    private DiseaseReportService diseaseReportService;

    @Autowired
    private UserService userService;

    @PostMapping
    public ResponseEntity<DiseaseReport> createReport(
            @RequestParam("images") List<MultipartFile> images,
            @RequestParam("description") String description,
            @AuthenticationPrincipal OAuth2User principal) throws IOException {

        if (principal == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        String email = principal.getAttribute("email");
        User user = userService.findByEmail(email).orElseThrow();

        DiseaseReport report = diseaseReportService.createReport(images, description, user);
        return ResponseEntity.ok(report);
    }

    @GetMapping("/my")
    public ResponseEntity<List<DiseaseReport>> getMyReports(@AuthenticationPrincipal OAuth2User principal) {
        if (principal == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        String email = principal.getAttribute("email");
        User user = userService.findByEmail(email).orElseThrow();
        List<DiseaseReport> reports = diseaseReportService.getReportsByUser(user.getUserId());
        return ResponseEntity.ok(reports);
    }

    @GetMapping("/all")
    public ResponseEntity<List<DiseaseReport>> getAllReports() {
        List<DiseaseReport> reports = diseaseReportService.getAllReports();
        return ResponseEntity.ok(reports);
    }

    @PutMapping("/{id}/status")
    public ResponseEntity<DiseaseReport> updateReportStatus(
            @PathVariable Long id,
            @RequestParam DiseaseReport.Status status,
            @RequestParam(required = false) String adminNotes) {
        DiseaseReport report = diseaseReportService.updateReportStatus(id, status, adminNotes);
        return ResponseEntity.ok(report);
    }
}