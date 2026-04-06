package com.farmerapp.backend.repository;

import com.farmerapp.backend.entity.DiseaseReport;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface DiseaseReportRepository extends JpaRepository<DiseaseReport, Long> {
    List<DiseaseReport> findByUser_UserId(Long userId);

    List<DiseaseReport> findByStatus(DiseaseReport.Status status);
}