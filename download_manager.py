"""
Download and Transfer Management System
Handles downloading results from Digital Ocean droplet to local machine
"""

import os
import json
import logging
import subprocess
import zipfile
import tarfile
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil


class DownloadManager:
    """Manages downloading and transferring scraping results"""
    
    def __init__(self, 
                 droplet_ip: str,
                 droplet_user: str = "root",
                 remote_project_dir: str = "/root/healthcare-scraper",
                 local_download_dir: str = "./downloads"):
        
        self.droplet_ip = droplet_ip
        self.droplet_user = droplet_user
        self.remote_project_dir = remote_project_dir
        self.local_download_dir = local_download_dir
        self.logger = logging.getLogger(__name__)
        
        # Create local download directory
        os.makedirs(local_download_dir, exist_ok=True)
    
    def create_download_package(self, package_name: str = None) -> str:
        """Create a compressed package of all results on the droplet"""
        
        if not package_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            package_name = f"healthcare_scraping_results_{timestamp}"
        
        # Commands to run on the droplet
        commands = [
            f"cd {self.remote_project_dir}",
            f"mkdir -p /tmp/{package_name}",
            
            # Copy all output files
            f"cp -r output/* /tmp/{package_name}/ 2>/dev/null || true",
            f"cp -r batch_output/* /tmp/{package_name}/ 2>/dev/null || true",
            
            # Copy logs
            f"mkdir -p /tmp/{package_name}/logs",
            f"cp -r logs/* /tmp/{package_name}/logs/ 2>/dev/null || true",
            
            # Copy configuration and input files
            f"cp corporate_chains.csv /tmp/{package_name}/ 2>/dev/null || true",
            f"cp input/* /tmp/{package_name}/ 2>/dev/null || true",
            
            # Create summary information
            f"echo 'Healthcare Facility Scraping Results' > /tmp/{package_name}/README.txt",
            f"echo 'Generated: {datetime.now().isoformat()}' >> /tmp/{package_name}/README.txt",
            f"echo 'Droplet IP: {self.droplet_ip}' >> /tmp/{package_name}/README.txt",
            f"echo '' >> /tmp/{package_name}/README.txt",
            f"echo 'Contents:' >> /tmp/{package_name}/README.txt",
            f"find /tmp/{package_name} -type f -name '*.json' -o -name '*.csv' | wc -l | xargs echo 'Data files:' >> /tmp/{package_name}/README.txt",
            f"du -sh /tmp/{package_name} | cut -f1 | xargs echo 'Total size:' >> /tmp/{package_name}/README.txt",
            
            # Create compressed archive
            f"cd /tmp && tar -czf {package_name}.tar.gz {package_name}/",
            f"ls -lh /tmp/{package_name}.tar.gz"
        ]
        
        # Execute commands on droplet
        full_command = " && ".join(commands)
        ssh_command = f"ssh {self.droplet_user}@{self.droplet_ip} '{full_command}'"
        
        try:
            result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully created package: {package_name}.tar.gz")
                return f"/tmp/{package_name}.tar.gz"
            else:
                self.logger.error(f"Failed to create package: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating package: {e}")
            return None
    
    def download_package(self, remote_package_path: str, local_filename: str = None) -> str:
        """Download the compressed package to local machine"""
        
        if not local_filename:
            local_filename = os.path.basename(remote_package_path)
        
        local_path = os.path.join(self.local_download_dir, local_filename)
        
        # Use SCP to download
        scp_command = f"scp {self.droplet_user}@{self.droplet_ip}:{remote_package_path} {local_path}"
        
        try:
            self.logger.info(f"Downloading {remote_package_path} to {local_path}")
            
            result = subprocess.run(scp_command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                self.logger.info(f"Successfully downloaded {local_filename} ({file_size_mb:.1f}MB)")
                return local_path
            else:
                self.logger.error(f"Failed to download package: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading package: {e}")
            return None
    
    def extract_package(self, package_path: str, extract_dir: str = None) -> str:
        """Extract the downloaded package"""
        
        if not extract_dir:
            extract_dir = os.path.join(self.local_download_dir, "extracted")
        
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            if package_path.endswith('.tar.gz'):
                with tarfile.open(package_path, 'r:gz') as tar:
                    tar.extractall(extract_dir)
            elif package_path.endswith('.zip'):
                with zipfile.ZipFile(package_path, 'r') as zip_file:
                    zip_file.extractall(extract_dir)
            else:
                self.logger.error(f"Unsupported package format: {package_path}")
                return None
            
            self.logger.info(f"Successfully extracted package to {extract_dir}")
            return extract_dir
            
        except Exception as e:
            self.logger.error(f"Error extracting package: {e}")
            return None
    
    def download_specific_files(self, file_patterns: List[str]) -> List[str]:
        """Download specific files matching patterns"""
        
        downloaded_files = []
        
        for pattern in file_patterns:
            # Find files matching pattern on droplet
            find_command = f"ssh {self.droplet_user}@{self.droplet_ip} 'find {self.remote_project_dir} -name \"{pattern}\" -type f'"
            
            try:
                result = subprocess.run(find_command, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    remote_files = result.stdout.strip().split('\n')
                    
                    for remote_file in remote_files:
                        if remote_file:  # Skip empty lines
                            local_filename = os.path.basename(remote_file)
                            local_path = os.path.join(self.local_download_dir, local_filename)
                            
                            # Download file
                            scp_command = f"scp {self.droplet_user}@{self.droplet_ip}:{remote_file} {local_path}"
                            download_result = subprocess.run(scp_command, shell=True, capture_output=True, text=True)
                            
                            if download_result.returncode == 0:
                                downloaded_files.append(local_path)
                                self.logger.info(f"Downloaded: {local_filename}")
                            else:
                                self.logger.error(f"Failed to download {remote_file}: {download_result.stderr}")
                
            except Exception as e:
                self.logger.error(f"Error downloading files with pattern {pattern}: {e}")
        
        return downloaded_files
    
    def sync_results_directory(self, local_sync_dir: str = None) -> str:
        """Sync entire results directory using rsync"""
        
        if not local_sync_dir:
            local_sync_dir = os.path.join(self.local_download_dir, "synced_results")
        
        os.makedirs(local_sync_dir, exist_ok=True)
        
        # Use rsync for efficient synchronization
        rsync_command = f"rsync -avz --progress {self.droplet_user}@{self.droplet_ip}:{self.remote_project_dir}/output/ {local_sync_dir}/"
        
        try:
            self.logger.info(f"Syncing results to {local_sync_dir}")
            
            result = subprocess.run(rsync_command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully synced results to {local_sync_dir}")
                return local_sync_dir
            else:
                self.logger.error(f"Failed to sync results: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error syncing results: {e}")
            return None
    
    def get_remote_file_list(self) -> List[Dict[str, Any]]:
        """Get list of available files on the droplet"""
        
        # Command to list files with details
        list_command = f"ssh {self.droplet_user}@{self.droplet_ip} 'find {self.remote_project_dir} -name \"*.json\" -o -name \"*.csv\" -o -name \"*.xlsx\" | xargs ls -lh'"
        
        try:
            result = subprocess.run(list_command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                files = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split()
                        if len(parts) >= 9:
                            files.append({
                                'permissions': parts[0],
                                'size': parts[4],
                                'date': f"{parts[5]} {parts[6]} {parts[7]}",
                                'path': parts[8],
                                'filename': os.path.basename(parts[8])
                            })
                
                return files
            else:
                self.logger.error(f"Failed to list remote files: {result.stderr}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error listing remote files: {e}")
            return []
    
    def cleanup_remote_files(self, older_than_days: int = 7) -> bool:
        """Clean up old files on the droplet"""
        
        cleanup_command = f"ssh {self.droplet_user}@{self.droplet_ip} 'find {self.remote_project_dir}/output -name \"*.json\" -o -name \"*.csv\" -mtime +{older_than_days} -delete'"
        
        try:
            result = subprocess.run(cleanup_command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully cleaned up files older than {older_than_days} days")
                return True
            else:
                self.logger.error(f"Failed to cleanup files: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cleaning up files: {e}")
            return False
    
    def create_download_summary(self, downloaded_files: List[str]) -> str:
        """Create a summary of downloaded files"""
        
        summary = {
            'download_date': datetime.now().isoformat(),
            'droplet_ip': self.droplet_ip,
            'total_files': len(downloaded_files),
            'files': []
        }
        
        total_size = 0
        
        for file_path in downloaded_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                summary['files'].append({
                    'filename': os.path.basename(file_path),
                    'path': file_path,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        
        summary['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        # Save summary
        summary_file = os.path.join(self.local_download_dir, 'download_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Download summary saved to {summary_file}")
        
        return summary_file


class AutoDownloader:
    """Automated download system with scheduling and monitoring"""
    
    def __init__(self, download_manager: DownloadManager):
        self.download_manager = download_manager
        self.logger = logging.getLogger(__name__)
    
    def setup_automated_download(self, 
                                schedule_type: str = "completion",
                                email_notification: str = None,
                                cloud_sync: str = None) -> bool:
        """Setup automated download when scraping completes"""
        
        # Create download script
        script_content = f"""#!/bin/bash

# Automated download script for healthcare facility scraper results
# Generated: {datetime.now().isoformat()}

DROPLET_IP="{self.download_manager.droplet_ip}"
DROPLET_USER="{self.download_manager.droplet_user}"
LOCAL_DIR="{self.download_manager.local_download_dir}"

echo "Starting automated download from $DROPLET_IP..."

# Create download package
ssh $DROPLET_USER@$DROPLET_IP 'cd /root/healthcare-scraper && python3 -c "
from download_manager import DownloadManager
dm = DownloadManager(\"localhost\")
package = dm.create_download_package()
print(f\"Package created: {{package}}\")
"'

# Download the package
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="healthcare_scraping_results_$TIMESTAMP.tar.gz"

scp $DROPLET_USER@$DROPLET_IP:/tmp/$PACKAGE_NAME $LOCAL_DIR/

# Extract package
cd $LOCAL_DIR
tar -xzf $PACKAGE_NAME

echo "Download completed: $LOCAL_DIR/$PACKAGE_NAME"

# Optional: Send email notification
{f'echo "Healthcare scraping results downloaded" | mail -s "Scraping Complete" {email_notification}' if email_notification else '# No email notification configured'}

# Optional: Sync to cloud storage
{f'# Add cloud sync commands here for {cloud_sync}' if cloud_sync else '# No cloud sync configured'}
"""
        
        script_path = os.path.join(self.download_manager.local_download_dir, 'auto_download.sh')
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        self.logger.info(f"Automated download script created: {script_path}")
        
        return True


# Usage examples and CLI interface

def main():
    """CLI interface for download management"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Healthcare Facility Scraper - Download Manager')
    
    parser.add_argument('--droplet-ip', required=True, help='Digital Ocean droplet IP address')
    parser.add_argument('--user', default='root', help='SSH user (default: root)')
    parser.add_argument('--download-dir', default='./downloads', help='Local download directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List files command
    list_parser = subparsers.add_parser('list', help='List available files on droplet')
    
    # Download all command
    download_all_parser = subparsers.add_parser('download-all', help='Download all results')
    download_all_parser.add_argument('--extract', action='store_true', help='Extract downloaded package')
    
    # Download specific command
    download_specific_parser = subparsers.add_parser('download-specific', help='Download specific files')
    download_specific_parser.add_argument('--patterns', nargs='+', required=True, help='File patterns to download')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync results directory')
    sync_parser.add_argument('--local-dir', help='Local sync directory')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old files on droplet')
    cleanup_parser.add_argument('--days', type=int, default=7, help='Delete files older than N days')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize download manager
    dm = DownloadManager(
        droplet_ip=args.droplet_ip,
        droplet_user=args.user,
        local_download_dir=args.download_dir
    )
    
    # Execute command
    if args.command == 'list':
        files = dm.get_remote_file_list()
        print(f"\nAvailable files on {args.droplet_ip}:")
        print("-" * 80)
        for file_info in files:
            print(f"{file_info['size']:>8} {file_info['date']:>20} {file_info['filename']}")
        print(f"\nTotal files: {len(files)}")
    
    elif args.command == 'download-all':
        package_path = dm.create_download_package()
        if package_path:
            local_path = dm.download_package(package_path)
            if local_path and args.extract:
                extract_dir = dm.extract_package(local_path)
                print(f"Results extracted to: {extract_dir}")
    
    elif args.command == 'download-specific':
        downloaded = dm.download_specific_files(args.patterns)
        print(f"Downloaded {len(downloaded)} files:")
        for file_path in downloaded:
            print(f"  {file_path}")
    
    elif args.command == 'sync':
        sync_dir = dm.sync_results_directory(args.local_dir)
        if sync_dir:
            print(f"Results synced to: {sync_dir}")
    
    elif args.command == 'cleanup':
        success = dm.cleanup_remote_files(args.days)
        if success:
            print(f"Cleaned up files older than {args.days} days")


if __name__ == '__main__':
    main()

