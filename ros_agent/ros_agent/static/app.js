/**
 * ROS2 Agent Dashboard JavaScript Client
 * 
 * Handles WebSocket communication, UI interactions, and real-time updates.
 */

class ROS2Dashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.currentSection = 'dashboard';
        this.chatHistory = [];
        this.logs = [];
        this.locations = {};
        this.lastUpdateTime = null;
        
        // Initialize the dashboard
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
        this.setupThemeToggle();
    }
    
    setupEventListeners() {
        // Navigation menu
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const section = e.currentTarget.dataset.section;
                this.switchSection(section);
            });
        });
        
        // Quick action buttons
        document.getElementById('stopNavigation')?.addEventListener('click', () => {
            this.sendNavigationCommand('stop');
        });
        
        document.getElementById('getStatus')?.addEventListener('click', () => {
            this.sendNavigationCommand('status');
        });
        
        document.getElementById('refreshData')?.addEventListener('click', () => {
            this.refreshData();
        });
        
        // Chat functionality
        document.getElementById('sendMessage')?.addEventListener('click', () => {
            this.sendChatMessage();
        });
        
        document.getElementById('chatInput')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });
        
        document.getElementById('clearChat')?.addEventListener('click', () => {
            this.clearChat();
        });
        
        // Logs functionality
        document.getElementById('logFilter')?.addEventListener('change', () => {
            this.filterLogs();
        });
        
        document.getElementById('logSearch')?.addEventListener('input', () => {
            this.filterLogs();
        });
        
        document.getElementById('exportLogs')?.addEventListener('click', () => {
            this.exportLogs();
        });
        
        // Locations functionality
        document.getElementById('addLocationBtn')?.addEventListener('click', () => {
            this.showAddLocationModal();
        });
        
        document.getElementById('closeAddLocationModal')?.addEventListener('click', () => {
            this.hideAddLocationModal();
        });
        
        document.getElementById('cancelAddLocation')?.addEventListener('click', () => {
            this.hideAddLocationModal();
        });
        
        document.getElementById('saveLocation')?.addEventListener('click', () => {
            this.saveLocation();
        });
        
        // Modal backdrop click
        document.getElementById('addLocationModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'addLocationModal') {
                this.hideAddLocationModal();
            }
        });
    }
    
    connectWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateConnectionStatus(true);
            this.showToast('Connected to ROS2 Agent', 'success');
            console.log('Connected to WebSocket server');
        });
        
        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus(false);
            this.showToast('Disconnected from ROS2 Agent', 'warning');
            console.log('Disconnected from WebSocket server');
        });
        
        this.socket.on('status_update', (data) => {
            this.updateDashboard(data);
        });
        
        this.socket.on('navigation_update', (data) => {
            this.updateNavigation(data);
        });
        
        this.socket.on('sensor_update', (data) => {
            this.updateSensors(data);
        });
        
        this.socket.on('log_entry', (data) => {
            this.addLogEntry(data);
        });
        
        this.socket.on('location_update', (data) => {
            this.updateLocations(data.locations);
        });
        
        this.socket.on('agent_response', (data) => {
            this.handleAgentResponse(data);
        });
        
        this.socket.on('navigation_result', (data) => {
            this.handleNavigationResult(data);
        });
        
        this.socket.on('error', (data) => {
            this.showToast(data.message || 'An error occurred', 'error');
        });
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        const icon = statusElement.querySelector('i');
        const text = statusElement.querySelector('span');
        
        if (connected) {
            statusElement.className = 'connection-status connected';
            text.textContent = 'Connected';
        } else {
            statusElement.className = 'connection-status disconnected';
            text.textContent = 'Disconnected';
        }
    }
    
    switchSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');
        
        // Update content sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionName).classList.add('active');
        
        this.currentSection = sectionName;
        
        // Load section-specific data
        if (sectionName === 'logs') {
            this.loadLogs();
        } else if (sectionName === 'locations') {
            this.loadLocations();
        }
    }
    
    updateDashboard(data) {
        // Update robot status
        document.getElementById('nodeName').textContent = data.node_name || '-';
        document.getElementById('subscriberCount').textContent = data.subscribers || '-';
        document.getElementById('serviceCount').textContent = data.services_discovered || '-';
        document.getElementById('aiAgentStatus').textContent = data.gemma_client_available ? 'Available' : 'Unavailable';
        document.getElementById('dataCacheCount').textContent = data.sensor_data_count || '-';
        document.getElementById('discoveryStatus').textContent = data.discovery_enabled ? 'Enabled' : 'Disabled';
        
        // Update last updated time
        this.lastUpdateTime = new Date();
        document.getElementById('dashboardLastUpdated').textContent = 
            `Last updated: ${this.lastUpdateTime.toLocaleTimeString()}`;
    }
    
    updateNavigation(data) {
        document.getElementById('navStatus').textContent = data.status || '-';
        document.getElementById('goalActive').textContent = data.goal_active ? 'Yes' : 'No';
        
        if (data.current_pose) {
            const pose = data.current_pose;
            document.getElementById('robotPosition').textContent = 
                `(${pose.x.toFixed(3)}, ${pose.y.toFixed(3)})`;
            document.getElementById('robotFrame').textContent = pose.frame_id || '-';
        } else {
            document.getElementById('robotPosition').textContent = 'Unknown';
            document.getElementById('robotFrame').textContent = '-';
        }
    }
    
    updateSensors(data) {
        // Update sensor status indicators
        const topics = data.topics || [];
        
        // Check for camera topics
        const hasCamera = topics.some(topic => topic.includes('camera') || topic.includes('image'));
        document.getElementById('cameraStatus').textContent = hasCamera ? 'Active' : 'Inactive';
        
        // Check for LiDAR topics
        const hasLidar = topics.some(topic => topic.includes('scan') || topic.includes('laser'));
        document.getElementById('lidarStatus').textContent = hasLidar ? 'Active' : 'Inactive';
        
        // Check for IMU topics
        const hasImu = topics.some(topic => topic.includes('imu'));
        document.getElementById('imuStatus').textContent = hasImu ? 'Active' : 'Inactive';
        
        // Check for odometry topics
        const hasOdom = topics.some(topic => topic.includes('odom'));
        document.getElementById('odomStatus').textContent = hasOdom ? 'Active' : 'Inactive';
    }
    
    sendNavigationCommand(command) {
        if (!this.isConnected) {
            this.showToast('Not connected to ROS2 Agent', 'error');
            return;
        }
        
        this.socket.emit('navigation_command', { command });
    }
    
    handleNavigationResult(data) {
        this.showToast(`Navigation ${data.command}: ${data.result}`, 'info');
    }
    
    sendChatMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        if (!this.isConnected) {
            this.showToast('Not connected to ROS2 Agent', 'error');
            return;
        }
        
        // Add user message to chat
        this.addChatMessage(message, 'user');
        
        // Clear input
        input.value = '';
        
        // Update chat status
        document.getElementById('chatStatus').textContent = 'Sending message...';
        
        // Send to agent
        this.socket.emit('agent_query', { question: message });
    }
    
    handleAgentResponse(data) {
        // Update chat status
        document.getElementById('chatStatus').textContent = 'Ready to chat';
        
        // Add agent response to chat
        this.addChatMessage(data.response, 'agent', data);
        
        // Show success/error toast
        if (data.success) {
            this.showToast('Agent response received', 'success');
        } else {
            this.showToast('Agent error: ' + (data.error || 'Unknown error'), 'error');
        }
    }
    
    addChatMessage(message, sender, metadata = null) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (sender === 'user') {
            contentDiv.innerHTML = `<i class="fas fa-user"></i><span>${this.escapeHtml(message)}</span>`;
        } else {
            let icon = 'fas fa-robot';
            if (metadata && !metadata.success) {
                icon = 'fas fa-exclamation-triangle';
            }
            contentDiv.innerHTML = `<i class="${icon}"></i><span>${this.escapeHtml(message)}</span>`;
            
            // Add metadata if available
            if (metadata && metadata.metadata) {
                const metaDiv = document.createElement('div');
                metaDiv.style.fontSize = '0.8em';
                metaDiv.style.marginTop = '0.5rem';
                metaDiv.style.opacity = '0.8';
                
                const usage = metadata.metadata.usage || {};
                const timings = metadata.metadata.timings || {};
                const metrics = metadata.metadata.metrics || {};
                
                metaDiv.innerHTML = `
                    <div>Response time: ${metadata.response_time?.toFixed(2)}s</div>
                    ${usage.prompt_tokens ? `<div>Tokens: ${usage.prompt_tokens} in, ${usage.completion_tokens} out</div>` : ''}
                    ${timings.prompt_ms ? `<div>Timing: ${timings.prompt_ms}ms prompt, ${timings.predicted_ms}ms generation</div>` : ''}
                    ${metrics.overall_tokens_per_second ? `<div>Speed: ${metrics.overall_tokens_per_second.toFixed(1)} tokens/s</div>` : ''}
                `;
                
                contentDiv.appendChild(metaDiv);
            }
        }
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Store in history
        this.chatHistory.push({
            message,
            sender,
            timestamp: new Date(),
            metadata
        });
    }
    
    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = `
            <div class="message system-message">
                <div class="message-content">
                    <i class="fas fa-robot"></i>
                    <span>Welcome! I'm your ROS2 agent assistant. Ask me anything about the robot's status, sensors, or navigation.</span>
                </div>
            </div>
        `;
        this.chatHistory = [];
    }
    
    addLogEntry(data) {
        this.logs.unshift(data);
        
        // Keep only last 1000 logs
        if (this.logs.length > 1000) {
            this.logs = this.logs.slice(0, 1000);
        }
        
        // Update logs table if currently viewing logs
        if (this.currentSection === 'logs') {
            this.updateLogsTable();
        }
    }
    
    loadLogs() {
        if (!this.isConnected) {
            this.showToast('Not connected to ROS2 Agent', 'error');
            return;
        }
        
        fetch('/api/logs')
            .then(response => response.json())
            .then(data => {
                this.logs = data.logs || [];
                this.updateLogsTable();
            })
            .catch(error => {
                console.error('Error loading logs:', error);
                this.showToast('Error loading logs', 'error');
            });
    }
    
    updateLogsTable() {
        const tbody = document.getElementById('logsTableBody');
        const filter = document.getElementById('logFilter').value;
        const search = document.getElementById('logSearch').value.toLowerCase();
        
        // Filter logs
        let filteredLogs = this.logs;
        
        if (filter !== 'all') {
            filteredLogs = filteredLogs.filter(log => {
                switch (filter) {
                    case 'success':
                        return log.success === true;
                    case 'error':
                        return log.success === false;
                    case 'navigation':
                        return log.navigation_command === true;
                    case 'service':
                        return log.service_call === true;
                    default:
                        return true;
                }
            });
        }
        
        if (search) {
            filteredLogs = filteredLogs.filter(log => 
                log.question.toLowerCase().includes(search) ||
                log.response.toLowerCase().includes(search)
            );
        }
        
        // Clear table
        tbody.innerHTML = '';
        
        if (filteredLogs.length === 0) {
            tbody.innerHTML = '<tr class="no-data"><td colspan="6">No logs match the current filter</td></tr>';
            return;
        }
        
        // Add log rows
        filteredLogs.forEach(log => {
            const row = document.createElement('tr');
            
            const timestamp = new Date(log.timestamp * 1000).toLocaleString();
            const question = this.truncateText(log.question, 50);
            const response = this.truncateText(log.response, 50);
            
            let tokensText = '-';
            if (log.tokens_input || log.tokens_output) {
                tokensText = `${log.tokens_input || 0}/${log.tokens_output || 0}`;
            }
            
            let timingText = '-';
            if (log.prompt_ms || log.predicted_ms) {
                timingText = `${log.prompt_ms || 0}ms/${log.predicted_ms || 0}ms`;
            }
            
            const statusIcon = log.success ? 'fas fa-check-circle text-success' : 'fas fa-times-circle text-danger';
            
            row.innerHTML = `
                <td>${timestamp}</td>
                <td title="${this.escapeHtml(log.question)}">${this.escapeHtml(question)}</td>
                <td title="${this.escapeHtml(log.response)}">${this.escapeHtml(response)}</td>
                <td>${tokensText}</td>
                <td>${timingText}</td>
                <td><i class="${statusIcon}"></i></td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    filterLogs() {
        this.updateLogsTable();
    }
    
    exportLogs() {
        if (this.logs.length === 0) {
            this.showToast('No logs to export', 'warning');
            return;
        }
        
        // Create CSV content
        const headers = [
            'Timestamp', 'Question', 'Response', 'Response Time', 'Processing Time',
            'Tokens Input', 'Tokens Output', 'Prompt MS', 'Predicted MS',
            'Tokens Per Second', 'Efficiency Score', 'Success', 'Error Message'
        ];
        
        const csvContent = [
            headers.join(','),
            ...this.logs.map(log => [
                new Date(log.timestamp * 1000).toISOString(),
                `"${log.question.replace(/"/g, '""')}"`,
                `"${log.response.replace(/"/g, '""')}"`,
                log.response_time || '',
                log.processing_time || '',
                log.tokens_input || '',
                log.tokens_output || '',
                log.prompt_ms || '',
                log.predicted_ms || '',
                log.tokens_per_second || '',
                log.efficiency_score || '',
                log.success || false,
                `"${(log.error_message || '').replace(/"/g, '""')}"`
            ].join(','))
        ].join('\n');
        
        // Download CSV
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ros_agent_logs_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showToast('Logs exported successfully', 'success');
    }
    
    loadLocations() {
        if (!this.isConnected) {
            this.showToast('Not connected to ROS2 Agent', 'error');
            return;
        }
        
        fetch('/api/locations')
            .then(response => response.json())
            .then(data => {
                this.locations = data.locations || {};
                this.updateLocationsTable();
            })
            .catch(error => {
                console.error('Error loading locations:', error);
                this.showToast('Error loading locations', 'error');
            });
    }
    
    updateLocations(locations) {
        this.locations = locations;
        this.updateLocationsTable();
    }
    
    updateLocationsTable() {
        const tbody = document.getElementById('locationsTableBody');
        
        // Clear table
        tbody.innerHTML = '';
        
        if (Object.keys(this.locations).length === 0) {
            tbody.innerHTML = '<tr class="no-data"><td colspan="5">No locations saved</td></tr>';
            return;
        }
        
        // Add location rows
        Object.entries(this.locations).forEach(([name, coords]) => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${this.escapeHtml(name)}</td>
                <td>${coords.x.toFixed(3)}</td>
                <td>${coords.y.toFixed(3)}</td>
                <td>${coords.yaw.toFixed(3)}</td>
                <td>
                    <button class="action-btn secondary" onclick="dashboard.navigateToLocation('${name}')">
                        <i class="fas fa-play"></i> Go
                    </button>
                    <button class="action-btn secondary" onclick="dashboard.removeLocation('${name}')">
                        <i class="fas fa-trash"></i> Delete
                    </button>
                </td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    showAddLocationModal() {
        document.getElementById('addLocationModal').classList.add('show');
        document.getElementById('locationName').focus();
    }
    
    hideAddLocationModal() {
        document.getElementById('addLocationModal').classList.remove('show');
        document.getElementById('addLocationForm').reset();
    }
    
    saveLocation() {
        const name = document.getElementById('locationName').value.trim();
        const x = parseFloat(document.getElementById('locationX').value);
        const y = parseFloat(document.getElementById('locationY').value);
        const yaw = parseFloat(document.getElementById('locationYaw').value) || 0;
        
        if (!name || isNaN(x) || isNaN(y)) {
            this.showToast('Please fill in all required fields', 'warning');
            return;
        }
        
        if (!this.isConnected) {
            this.showToast('Not connected to ROS2 Agent', 'error');
            return;
        }
        
        fetch('/api/locations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ label: name, x, y, yaw })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                this.showToast('Error: ' + data.error, 'error');
            } else {
                this.showToast(data.message, 'success');
                this.hideAddLocationModal();
                this.loadLocations();
            }
        })
        .catch(error => {
            console.error('Error saving location:', error);
            this.showToast('Error saving location', 'error');
        });
    }
    
    navigateToLocation(name) {
        if (!this.isConnected) {
            this.showToast('Not connected to ROS2 Agent', 'error');
            return;
        }
        
        // Send navigation command via chat
        const message = `move_to_position(${name})`;
        this.addChatMessage(message, 'user');
        
        this.socket.emit('agent_query', { question: message });
    }
    
    removeLocation(name) {
        if (!confirm(`Are you sure you want to delete the location "${name}"?`)) {
            return;
        }
        
        if (!this.isConnected) {
            this.showToast('Not connected to ROS2 Agent', 'error');
            return;
        }
        
        fetch(`/api/locations/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                this.showToast('Error: ' + data.error, 'error');
            } else {
                this.showToast(data.message, 'success');
                this.loadLocations();
            }
        })
        .catch(error => {
            console.error('Error removing location:', error);
            this.showToast('Error removing location', 'error');
        });
    }
    
    refreshData() {
        this.loadInitialData();
        this.showToast('Data refreshed', 'success');
    }
    
    loadInitialData() {
        if (!this.isConnected) return;
        
        // Load logs if on logs section
        if (this.currentSection === 'logs') {
            this.loadLogs();
        }
        
        // Load locations if on locations section
        if (this.currentSection === 'locations') {
            this.loadLocations();
        }
    }
    
    setupThemeToggle() {
        const themeToggle = document.getElementById('themeToggle');
        const currentTheme = localStorage.getItem('theme') || 'light';
        
        // Apply saved theme
        document.documentElement.setAttribute('data-theme', currentTheme);
        this.updateThemeIcon(currentTheme);
        
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            this.updateThemeIcon(newTheme);
        });
    }
    
    updateThemeIcon(theme) {
        const icon = document.querySelector('#themeToggle i');
        icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
    
    showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const iconMap = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        toast.innerHTML = `
            <i class="${iconMap[type]}"></i>
            <div class="toast-content">
                <div class="toast-message">${this.escapeHtml(message)}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        container.appendChild(toast);
        
        // Auto remove after duration
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, duration);
    }
    
    showLoading(show = true) {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ROS2Dashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause updates
        console.log('Page hidden, pausing updates');
    } else {
        // Page is visible, resume updates
        console.log('Page visible, resuming updates');
        if (window.dashboard) {
            window.dashboard.loadInitialData();
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    // Handle responsive layout changes if needed
});

// Handle errors
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.dashboard) {
        window.dashboard.showToast('An unexpected error occurred', 'error');
    }
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.dashboard) {
        window.dashboard.showToast('An unexpected error occurred', 'error');
    }
});
