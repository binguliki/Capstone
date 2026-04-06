package com.farmerapp.backend.controller;

import com.farmerapp.backend.entity.User;
import com.farmerapp.backend.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class AuthController {

    @Autowired
    private UserService userService;

    @GetMapping("/user")
    public ResponseEntity<User> getUser(@AuthenticationPrincipal OAuth2User principal) {
        if (principal == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        String email = principal.getAttribute("email");
        User user = userService.findByEmail(email).orElseGet(() -> {
            // Create user if not exists, default to FARMER
            return userService.createUser(
                    principal.getAttribute("name"),
                    email,
                    null, // phone
                    "", // no password for OAuth
                    User.Role.FARMER);
        });
        return ResponseEntity.ok(user);
    }
}