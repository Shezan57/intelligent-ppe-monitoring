/**
 * Settings Panel Component
 * 
 * Configure detection thresholds and system settings.
 */

import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'

// Default settings
const DEFAULT_SETTINGS = {
    // Detection thresholds
    yolo_confidence: 0.25,
    sam_mask_threshold: 0.05,

    // Processing options
    frame_skip: 5,
    save_annotated: true,
    sam_enabled: false,

    // Site info
    site_location: 'Construction Site A',
    camera_id: 'CAM-001'
}

function SettingsPanel({ onClose }) {
    const [settings, setSettings] = useState(DEFAULT_SETTINGS)
    const [isSaving, setIsSaving] = useState(false)

    // Load settings from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem('ppe_settings')
        if (saved) {
            try {
                setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(saved) })
            } catch (e) {
                console.error('Failed to load settings:', e)
            }
        }
    }, [])

    // Handle input changes
    const handleChange = (key, value) => {
        setSettings(prev => ({ ...prev, [key]: value }))
    }

    // Save settings
    const handleSave = () => {
        setIsSaving(true)

        try {
            localStorage.setItem('ppe_settings', JSON.stringify(settings))
            toast.success('Settings saved successfully')
            onClose?.()
        } catch (e) {
            toast.error('Failed to save settings')
        } finally {
            setIsSaving(false)
        }
    }

    // Reset to defaults
    const handleReset = () => {
        setSettings(DEFAULT_SETTINGS)
        localStorage.removeItem('ppe_settings')
        toast.success('Settings reset to defaults')
    }

    return (
        <div className="settings-overlay" onClick={(e) => e.target === e.currentTarget && onClose?.()}>
            <div className="settings-panel card">
                <div className="settings-header">
                    <h3>⚙️ Settings</h3>
                    <button className="btn-icon" onClick={onClose}>✕</button>
                </div>

                <div className="settings-content">
                    {/* Detection Thresholds */}
                    <section className="settings-section">
                        <h4>Detection Thresholds</h4>

                        <div className="setting-item">
                            <label>YOLO Confidence</label>
                            <div className="setting-control">
                                <input
                                    type="range"
                                    min="0.1"
                                    max="0.9"
                                    step="0.05"
                                    value={settings.yolo_confidence}
                                    onChange={(e) => handleChange('yolo_confidence', parseFloat(e.target.value))}
                                />
                                <span className="setting-value">{settings.yolo_confidence.toFixed(2)}</span>
                            </div>
                            <small>Minimum confidence for YOLO detections</small>
                        </div>

                        <div className="setting-item">
                            <label>SAM Mask Threshold</label>
                            <div className="setting-control">
                                <input
                                    type="range"
                                    min="0.01"
                                    max="0.2"
                                    step="0.01"
                                    value={settings.sam_mask_threshold}
                                    onChange={(e) => handleChange('sam_mask_threshold', parseFloat(e.target.value))}
                                />
                                <span className="setting-value">{(settings.sam_mask_threshold * 100).toFixed(0)}%</span>
                            </div>
                            <small>Minimum mask coverage for SAM verification</small>
                        </div>
                    </section>

                    {/* Processing Options */}
                    <section className="settings-section">
                        <h4>Processing Options</h4>

                        <div className="setting-item">
                            <label>SAM Verification</label>
                            <div className="setting-control">
                                <label className="toggle">
                                    <input
                                        type="checkbox"
                                        checked={settings.sam_enabled}
                                        onChange={(e) => handleChange('sam_enabled', e.target.checked)}
                                    />
                                    <span className="toggle-slider"></span>
                                </label>
                                <span className="setting-value">
                                    {settings.sam_enabled ? 'Enabled' : 'Disabled (GPU required)'}
                                </span>
                            </div>
                            <small>Enable SAM semantic verification (requires GPU)</small>
                        </div>

                        <div className="setting-item">
                            <label>Video Frame Skip</label>
                            <div className="setting-control">
                                <input
                                    type="range"
                                    min="1"
                                    max="15"
                                    step="1"
                                    value={settings.frame_skip}
                                    onChange={(e) => handleChange('frame_skip', parseInt(e.target.value))}
                                />
                                <span className="setting-value">{settings.frame_skip}</span>
                            </div>
                            <small>Process every Nth frame (higher = faster, less accurate)</small>
                        </div>

                        <div className="setting-item">
                            <label>Save Annotated Images</label>
                            <div className="setting-control">
                                <label className="toggle">
                                    <input
                                        type="checkbox"
                                        checked={settings.save_annotated}
                                        onChange={(e) => handleChange('save_annotated', e.target.checked)}
                                    />
                                    <span className="toggle-slider"></span>
                                </label>
                                <span className="setting-value">
                                    {settings.save_annotated ? 'Yes' : 'No'}
                                </span>
                            </div>
                        </div>
                    </section>

                    {/* Site Configuration */}
                    <section className="settings-section">
                        <h4>Site Configuration</h4>

                        <div className="setting-item">
                            <label>Site Location</label>
                            <input
                                type="text"
                                className="input"
                                value={settings.site_location}
                                onChange={(e) => handleChange('site_location', e.target.value)}
                            />
                        </div>

                        <div className="setting-item">
                            <label>Camera ID</label>
                            <input
                                type="text"
                                className="input"
                                value={settings.camera_id}
                                onChange={(e) => handleChange('camera_id', e.target.value)}
                            />
                        </div>
                    </section>
                </div>

                <div className="settings-footer">
                    <button className="btn btn--secondary" onClick={handleReset}>
                        Reset Defaults
                    </button>
                    <button
                        className="btn btn--primary"
                        onClick={handleSave}
                        disabled={isSaving}
                    >
                        {isSaving ? 'Saving...' : 'Save Settings'}
                    </button>
                </div>
            </div>
        </div>
    )
}

export default SettingsPanel
