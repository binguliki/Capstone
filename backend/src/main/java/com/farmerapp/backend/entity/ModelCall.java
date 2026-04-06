package com.farmerapp.backend.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Entity
@Table(name = "model_calls")
@Data
public class ModelCall {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long modelCallId;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;

    @Column(length = 1024)
    private String imageUrl;
    private String predictionLabel;
    private Double confidenceScore;

    @Column(columnDefinition = "TEXT")
    private String responseJson;

    @Enumerated(EnumType.STRING)
    private Status status;

    private LocalDateTime createdAt;

    public enum Status {
        SUCCESS, FAILED, LOW_CONFIDENCE
    }
}