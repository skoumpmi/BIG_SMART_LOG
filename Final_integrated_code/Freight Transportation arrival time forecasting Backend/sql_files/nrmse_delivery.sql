-- --------------------------------------------------------
-- Host:                         127.0.0.1
-- Server version:               8.0.23 - MySQL Community Server - GPL
-- Server OS:                    Win64
-- HeidiSQL Version:             11.2.0.6213
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;


-- Dumping database structure for bigsmartlog
CREATE DATABASE IF NOT EXISTS `bigsmartlog` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `bigsmartlog`;
CREATE TABLE bigsmartlog.nrmse_delivery(id VARCHAR(50) NOT NULL,WH1 VARCHAR(50) NOT NULL, WH2 VARCHAR(50) NOT NULL,WH3 VARCHAR(50) NOT NULL,WH4 VARCHAR(50) NOT NULL,WH5 VARCHAR(50) NOT NULL,PRIMARY KEY(id)); 


INSERT INTO bigsmartlog.nrmse_delivery(id,WH1,WH2,WH3,WH4,WH5) VALUES ('RF',"0.9051","0.7862","0.9253","0.7895","0.8361");
INSERT INTO bigsmartlog.nrmse_delivery(id,WH1,WH2,WH3,WH4,WH5) VALUES ('B',"0.9087","0.7985","0.7706","0.8597","0.9155");
INSERT INTO bigsmartlog.nrmse_delivery(id,WH1,WH2,WH3,WH4,WH5) VALUES ('GB',"0.9229","0.8646","0.9121","0.8766","0.9670");
INSERT INTO bigsmartlog.nrmse_delivery(id,WH1,WH2,WH3,WH4,WH5) VALUES ('WN',"0.9592","0.970","0.9939","0.946","0.9803");

-- Data exporting was unselected.

/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IFNULL(@OLD_FOREIGN_KEY_CHECKS, 1) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40111 SET SQL_NOTES=IFNULL(@OLD_SQL_NOTES, 1) */;
