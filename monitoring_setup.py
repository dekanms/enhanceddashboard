# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:58:02 2024

@author: Adeka
"""

# monitoring_setup.py

import os
import json
import shutil
import logging
from datetime import datetime
import torch
import numpy as np
from logging.handlers import RotatingFileHandler


class MonitoringSystem:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.backups_dir = os.path.join(self.base_dir, 'backups')
        self.loggers = {}
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            os.path.join(self.logs_dir),
            os.path.join(self.logs_dir, 'archive'),
            os.path.join(self.backups_dir),
            os.path.join(self.backups_dir, 'models'),
            os.path.join(self.backups_dir, 'metrics'),
            os.path.join(self.backups_dir, 'configs')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def get_logger(self, name):
        """Get or create a logger with the given name"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            
            # Create file handler
            log_file = os.path.join(self.logs_dir, f'{name}.log')
            handler = logging.FileHandler(log_file)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(handler)
            self.loggers[name] = logger
            
        return self.loggers[name]


    def setup_logging(self):
        """Setup logging configuration"""
        log_files = {
            'api': 'api_logs.log',
            'deployment': 'deployment.log',
            'metrics': 'model_metrics.log',
            'system': 'system_status.log'
        }
        
        for log_name, log_file in log_files.items():
            logger = logging.getLogger(log_name)
            logger.setLevel(logging.INFO)
            
            handler = RotatingFileHandler(
                os.path.join(self.logs_dir, log_file),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
    def backup_model(self, model_path='production_model.pth'):
        """Create backup of model"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(
                self.backups_dir, 
                'models', 
                f'model_backup_{timestamp}.pth'
            )
            
            shutil.copy2(model_path, backup_path)
            logging.getLogger('system').info(f'Model backed up to {backup_path}')
            return True
        except Exception as e:
            logging.getLogger('system').error(f'Model backup failed: {str(e)}')
            return False
            
    def save_metrics(self, metrics):
        """Save performance metrics"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_path = os.path.join(
                self.backups_dir,
                'metrics',
                f'metrics_{timestamp}.json'
            )
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            logging.getLogger('metrics').info(f'Metrics saved to {metrics_path}')
            return True
        except Exception as e:
            logging.getLogger('metrics').error(f'Metrics save failed: {str(e)}')
            return False
            
    def save_config(self, config, config_type):
        """Save configuration backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            config_path = os.path.join(
                self.backups_dir,
                'configs',
                f'{config_type}_config_{timestamp}.json'
            )
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            logging.getLogger('system').info(f'Config saved to {config_path}')
            return True
        except Exception as e:
            logging.getLogger('system').error(f'Config save failed: {str(e)}')
            return False
            
    def cleanup_old_files(self, max_age_days=30):
        """Clean up old log and backup files"""
        try:
            current_time = datetime.now()
            
            # Clean up logs
            for root, _, files in os.walk(self.logs_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if (current_time - file_time).days > max_age_days:
                        os.remove(file_path)
                        
            # Clean up backups
            for root, _, files in os.walk(self.backups_dir):
                for file in files:
                    if file.startswith('model_backup_') or file.startswith('metrics_'):
                        file_path = os.path.join(root, file)
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if (current_time - file_time).days > max_age_days:
                            os.remove(file_path)
                            
            logging.getLogger('system').info('Cleanup completed successfully')
            return True
        except Exception as e:
            logging.getLogger('system').error(f'Cleanup failed: {str(e)}')
            return False

def main():
    # Initialize monitoring system
    monitor = MonitoringSystem()
    
    # Initial configurations
    api_config = {
        'version': '1.0',
        'host': '0.0.0.0',
        'port': 8000,
        'model_path': 'production_model.pth',
        'log_level': 'INFO'
    }
    
    dashboard_config = {
        'refresh_rate': 30,
        'max_history': 1000,
        'drift_threshold': 0.1
    }
    
    # Save initial configurations
    monitor.save_config(api_config, 'api')
    monitor.save_config(dashboard_config, 'dashboard')
    
    # Backup current model if exists
    if os.path.exists('production_model.pth'):
        monitor.backup_model()
    
    print("Monitoring system initialized successfully!")

if __name__ == "__main__":
    main()