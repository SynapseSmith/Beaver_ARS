-- ================================
-- Beaver ARS Database Schema
-- MySQL 8.0+
-- ================================

-- Create database
CREATE DATABASE IF NOT EXISTS beaver_ars
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE beaver_ars;

-- ================================
-- Tables
-- ================================

-- Menu Table
CREATE TABLE IF NOT EXISTS menus (
  menu_id INT AUTO_INCREMENT PRIMARY KEY,
  menu_name VARCHAR(255) NOT NULL,
  category VARCHAR(100) NOT NULL,
  price DECIMAL(10,2) NOT NULL,
  description TEXT,
  is_available BOOLEAN DEFAULT TRUE,
  image_url VARCHAR(500),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  INDEX idx_category (category),
  INDEX idx_menu_name (menu_name),
  INDEX idx_is_available (is_available),
  FULLTEXT INDEX ft_menu_name (menu_name),
  FULLTEXT INDEX ft_description (description)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Orders Table
CREATE TABLE IF NOT EXISTS orders (
  order_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(100) NOT NULL,
  menu_id INT NOT NULL,
  quantity INT NOT NULL DEFAULT 1,
  total_price DECIMAL(10,2) NOT NULL,
  order_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  status VARCHAR(50) DEFAULT 'pending',
  delivery_address TEXT,
  contact_number VARCHAR(20),
  notes TEXT,
  
  INDEX idx_user_id (user_id),
  INDEX idx_order_time (order_time),
  INDEX idx_status (status),
  
  FOREIGN KEY (menu_id) REFERENCES menus(menu_id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Conversations Table
CREATE TABLE IF NOT EXISTS conversations (
  conversation_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(100) NOT NULL,
  session_id VARCHAR(100),
  user_message TEXT NOT NULL,
  intent VARCHAR(100),
  confidence_score FLOAT,
  entities JSON,
  bot_response TEXT,
  response_time_ms INT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_user_id (user_id),
  INDEX idx_timestamp (timestamp),
  INDEX idx_intent (intent),
  INDEX idx_session_id (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- FAQs Table
CREATE TABLE IF NOT EXISTS faqs (
  faq_id INT AUTO_INCREMENT PRIMARY KEY,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  category VARCHAR(100),
  views INT DEFAULT 0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  is_active BOOLEAN DEFAULT TRUE,
  
  INDEX idx_category (category),
  INDEX idx_views (views),
  FULLTEXT INDEX ft_question (question),
  FULLTEXT INDEX ft_answer (answer)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Users Table
CREATE TABLE IF NOT EXISTS users (
  user_id VARCHAR(100) PRIMARY KEY,
  username VARCHAR(100),
  email VARCHAR(255),
  phone VARCHAR(20),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  preferences JSON,
  
  INDEX idx_email (email),
  INDEX idx_phone (phone)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Feedback Table
CREATE TABLE IF NOT EXISTS feedback (
  feedback_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(100),
  conversation_id INT,
  rating INT CHECK (rating BETWEEN 1 AND 5),
  comment TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_user_id (user_id),
  INDEX idx_rating (rating),
  
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
  FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System Logs Table
CREATE TABLE IF NOT EXISTS system_logs (
  log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  log_level VARCHAR(20),
  component VARCHAR(100),
  message TEXT,
  error_trace TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_log_level (log_level),
  INDEX idx_component (component),
  INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ================================
-- Sample Data
-- ================================

-- Insert sample menus
INSERT INTO menus (menu_name, category, price, description) VALUES
('김치찌개', '찌개류', 8000, '묵은지로 끓인 진한 김치찌개'),
('된장찌개', '찌개류', 7000, '구수한 된장찌개'),
('불고기', '고기류', 15000, '양념에 재운 소고기 불고기'),
('비빔밥', '밥류', 9000, '각종 나물과 고추장이 들어간 비빔밥'),
('냉면', '면류', 10000, '시원한 물냉면'),
('제육볶음', '고기류', 12000, '매콤한 제육볶음'),
('김치볶음밥', '밥류', 8000, '김치와 함께 볶은 볶음밥'),
('순두부찌개', '찌개류', 7500, '부드러운 순두부찌개');

-- Insert sample FAQs
INSERT INTO faqs (question, answer, category) VALUES
('영업시간이 어떻게 되나요?', '평일 오전 11시부터 오후 10시까지 영업합니다. 주말은 오전 10시부터 영업합니다.', '영업정보'),
('배달이 가능한가요?', '네, 배달 서비스를 제공하고 있습니다. 최소 주문 금액은 15,000원입니다.', '배달'),
('단체 예약이 가능한가요?', '네, 10인 이상 단체 예약이 가능합니다. 미리 연락 주시면 자리를 준비해드립니다.', '예약'),
('주차 공간이 있나요?', '네, 매장 앞 전용 주차장을 이용하실 수 있습니다.', '편의시설');

-- ================================
-- Views
-- ================================

-- Popular menu view
CREATE OR REPLACE VIEW popular_menus AS
SELECT 
  m.menu_id,
  m.menu_name,
  m.category,
  m.price,
  COUNT(o.order_id) as order_count,
  SUM(o.total_price) as total_revenue
FROM menus m
LEFT JOIN orders o ON m.menu_id = o.menu_id
WHERE o.order_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY m.menu_id
ORDER BY order_count DESC;

-- User activity view
CREATE OR REPLACE VIEW user_activity AS
SELECT 
  u.user_id,
  u.username,
  COUNT(DISTINCT c.conversation_id) as conversation_count,
  COUNT(DISTINCT o.order_id) as order_count,
  MAX(u.last_active) as last_active
FROM users u
LEFT JOIN conversations c ON u.user_id = c.user_id
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id;

-- ================================
-- Stored Procedures
-- ================================

DELIMITER //

-- Get menu recommendations
CREATE PROCEDURE GetMenuRecommendations(IN user_id_param VARCHAR(100))
BEGIN
  SELECT 
    m.menu_id,
    m.menu_name,
    m.category,
    m.price,
    COUNT(o.order_id) as popularity_score
  FROM menus m
  LEFT JOIN orders o ON m.menu_id = o.menu_id
  WHERE m.is_available = TRUE
  GROUP BY m.menu_id
  ORDER BY popularity_score DESC
  LIMIT 10;
END //

-- Create order
CREATE PROCEDURE CreateOrder(
  IN user_id_param VARCHAR(100),
  IN menu_id_param INT,
  IN quantity_param INT,
  IN delivery_address_param TEXT,
  IN contact_number_param VARCHAR(20),
  OUT order_id_out INT
)
BEGIN
  DECLARE menu_price DECIMAL(10,2);
  DECLARE total DECIMAL(10,2);
  
  -- Get menu price
  SELECT price INTO menu_price FROM menus WHERE menu_id = menu_id_param;
  SET total = menu_price * quantity_param;
  
  -- Insert order
  INSERT INTO orders (user_id, menu_id, quantity, total_price, delivery_address, contact_number)
  VALUES (user_id_param, menu_id_param, quantity_param, total, delivery_address_param, contact_number_param);
  
  SET order_id_out = LAST_INSERT_ID();
END //

DELIMITER ;

-- ================================
-- Triggers
-- ================================

DELIMITER //

-- Update FAQ views count
CREATE TRIGGER update_faq_views
AFTER INSERT ON conversations
FOR EACH ROW
BEGIN
  IF NEW.intent = 'faq_inquiry' THEN
    UPDATE faqs SET views = views + 1 WHERE faq_id IN (
      SELECT JSON_EXTRACT(NEW.entities, '$.faq_id')
    );
  END IF;
END //

DELIMITER ;

-- ================================
-- Indexes for Performance
-- ================================

-- Composite indexes for common queries
CREATE INDEX idx_orders_user_time ON orders(user_id, order_time);
CREATE INDEX idx_conversations_user_time ON conversations(user_id, timestamp);

-- ================================
-- Permissions
-- ================================

-- Create application user
CREATE USER IF NOT EXISTS 'beaver_user'@'%' IDENTIFIED BY 'beaver_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON beaver_ars.* TO 'beaver_user'@'%';
GRANT EXECUTE ON beaver_ars.* TO 'beaver_user'@'%';
FLUSH PRIVILEGES;
