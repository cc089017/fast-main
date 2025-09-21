-- 전체 DB 생성 SQL (user, face, arm, speech)

CREATE TABLE IF NOT EXISTS `user` (
  id            VARCHAR(50)  PRIMARY KEY,
  email         VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  name          VARCHAR(100) NOT NULL,
  birth_date    DATE         NOT NULL,
  phone_number  VARCHAR(20)  NULL,
  gender        ENUM('male','female') NULL,
  privacy_agreed TINYINT(1)  NOT NULL DEFAULT 0,
  created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_user_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `face` (
  `face_id`        BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `user_id`        VARCHAR(50)     NOT NULL,
  `image_blob`     LONGBLOB        NOT NULL,
  `image_mime`     VARCHAR(64)     NOT NULL DEFAULT 'image/jpeg',
  `image_size`     INT UNSIGNED    NULL,
  `result_text`    TEXT            NOT NULL,
  `landmarks_json` JSON            NULL,
  `created_at`     DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at`     DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`face_id`),
  KEY `idx_face_user_created` (`user_id`, `created_at`),
  CONSTRAINT `fk_face_user`
    FOREIGN KEY (`user_id`) REFERENCES `user`(`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `arm` (
  `arm_id`           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `user_id`          VARCHAR(50)     NULL,
  `start_image_blob` LONGBLOB        NOT NULL,
  `start_image_mime` VARCHAR(64)     NOT NULL DEFAULT 'image/png',
  `start_image_size` INT UNSIGNED    NULL,
  `end_image_blob`   LONGBLOB        NOT NULL,
  `end_image_mime`   VARCHAR(64)     NOT NULL DEFAULT 'image/png',
  `end_image_size`   INT UNSIGNED    NULL,
  `label`            VARCHAR(64)     NULL,
  `confidence`       FLOAT           NULL,
  `features_json`    JSON            NULL,
  `created_at`       DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at`       DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`arm_id`),
  KEY `idx_arm_user_created` (`user_id`, `created_at`),
  CONSTRAINT `fk_arm_user`
    FOREIGN KEY (`user_id`) REFERENCES `user`(`id`)
    ON DELETE SET NULL
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `speech` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    wav_filename VARCHAR(255),
    model_tag VARCHAR(64),
    risk_score FLOAT,
    threshold FLOAT,
    result_text VARCHAR(32),
    dtw_slice1 FLOAT,
    dtw_slice2 FLOAT,
    dtw_slice3 FLOAT,
    dtw_slice4 FLOAT,
    dtw_slice5 FLOAT,
    dtw_slope FLOAT,
    x_cover FLOAT,
    voiced_sec FLOAT,
    dtw_graph_url VARCHAR(255),
    feature_graph_url VARCHAR(255),
    waveform_graph_url VARCHAR(255),
    INDEX(user_id),
    INDEX(created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
