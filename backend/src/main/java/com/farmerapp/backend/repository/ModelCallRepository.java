package com.farmerapp.backend.repository;

import com.farmerapp.backend.entity.ModelCall;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface ModelCallRepository extends JpaRepository<ModelCall, Long> {
    List<ModelCall> findByUser_UserId(Long userId);
}