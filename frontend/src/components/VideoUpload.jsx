/**
 * Video Upload Component
 * 
 * Upload and process video files for PPE detection.
 */

import { useState, useRef, useCallback } from 'react'
import toast from 'react-hot-toast'
import api from '../api/client'

const ALLOWED_TYPES = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm', 'video/x-matroska']
const MAX_SIZE = 100 * 1024 * 1024 // 100MB

function VideoUpload() {
    const [selectedFile, setSelectedFile] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [progress, setProgress] = useState(0)
    const [result, setResult] = useState(null)
    const [frameSkip, setFrameSkip] = useState(5)
    const fileInputRef = useRef(null)

    // Handle file selection
    const handleFileSelect = useCallback((file) => {
        if (!file) return

        // Validate type
        if (!ALLOWED_TYPES.includes(file.type)) {
            toast.error('Invalid file type. Use MP4, AVI, MOV, WEBM, or MKV')
            return
        }

        // Validate size
        if (file.size > MAX_SIZE) {
            toast.error('File too large (max 100MB)')
            return
        }

        setSelectedFile(file)
        setResult(null)
    }, [])

    // Handle drop
    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()

        const file = e.dataTransfer.files[0]
        handleFileSelect(file)
    }, [handleFileSelect])

    // Process video
    const handleProcess = async () => {
        if (!selectedFile) {
            toast.error('Please select a video first')
            return
        }

        setIsProcessing(true)
        setProgress(0)
        const loadingToast = toast.loading('Processing video...')

        try {
            const formData = new FormData()
            formData.append('file', selectedFile)
            formData.append('frame_skip', frameSkip)

            const response = await api.post('/detect/video', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 300000, // 5 minutes for long videos
                onUploadProgress: (e) => {
                    const pct = Math.round((e.loaded / e.total) * 50)
                    setProgress(pct)
                }
            })

            setProgress(100)
            setResult(response.data)

            const stats = response.data.aggregated_stats
            toast.success(
                `Processed! ${stats.total_violations} violations in ${stats.total_persons_detected} detections`,
                { id: loadingToast }
            )

        } catch (error) {
            console.error('Video processing error:', error)
            toast.error(
                error.response?.data?.detail || 'Video processing failed',
                { id: loadingToast }
            )
        } finally {
            setIsProcessing(false)
        }
    }

    // Clear selection
    const handleClear = () => {
        setSelectedFile(null)
        setResult(null)
        setProgress(0)
        if (fileInputRef.current) {
            fileInputRef.current.value = ''
        }
    }

    return (
        <div className="video-upload-container">
            {/* Upload Zone */}
            <div
                className={`upload-zone ${selectedFile ? 'upload-zone--has-file' : ''}`}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".mp4,.avi,.mov,.webm,.mkv"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFileSelect(e.target.files[0])}
                />

                {selectedFile ? (
                    <div className="upload-zone__preview">
                        <span className="upload-zone__icon">🎬</span>
                        <p className="upload-zone__filename">{selectedFile.name}</p>
                        <p className="upload-zone__size">
                            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                    </div>
                ) : (
                    <div className="upload-zone__placeholder">
                        <span className="upload-zone__icon">📹</span>
                        <p>Drop video file or click to select</p>
                        <small>MP4, AVI, MOV, WEBM (max 100MB)</small>
                    </div>
                )}
            </div>

            {/* Frame Skip Setting */}
            <div className="card" style={{ marginTop: '1rem' }}>
                <div className="setting-item">
                    <label>Frame Skip (process every Nth frame)</label>
                    <div className="setting-control">
                        <input
                            type="range"
                            min="1"
                            max="15"
                            value={frameSkip}
                            onChange={(e) => setFrameSkip(parseInt(e.target.value))}
                            disabled={isProcessing}
                        />
                        <span className="setting-value">{frameSkip}</span>
                    </div>
                    <small>Higher = faster processing, lower accuracy</small>
                </div>
            </div>

            {/* Progress Bar */}
            {isProcessing && (
                <div className="progress-container">
                    <div className="progress-bar">
                        <div
                            className="progress-bar__fill"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                    <span className="progress-text">{progress}%</span>
                </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-md" style={{ marginTop: '1rem' }}>
                <button
                    className="btn btn--primary btn--lg"
                    style={{ flex: 1 }}
                    onClick={handleProcess}
                    disabled={!selectedFile || isProcessing}
                >
                    {isProcessing ? (
                        <>
                            <span className="loading-spinner" style={{ width: '16px', height: '16px' }}></span>
                            Processing...
                        </>
                    ) : (
                        '🎬 Process Video'
                    )}
                </button>

                {selectedFile && (
                    <button
                        className="btn btn--secondary btn--lg"
                        onClick={handleClear}
                        disabled={isProcessing}
                    >
                        Clear
                    </button>
                )}
            </div>

            {/* Results */}
            {result && (
                <div className="card video-results" style={{ marginTop: '1rem' }}>
                    <h4>📊 Video Analysis Results</h4>

                    <div className="result-grid">
                        <div className="result-item">
                            <span className="result-label">Duration</span>
                            <span className="result-value">
                                {result.video_info?.duration_seconds?.toFixed(1)}s
                            </span>
                        </div>
                        <div className="result-item">
                            <span className="result-label">Frames Processed</span>
                            <span className="result-value">
                                {result.processing_info?.frames_processed}
                            </span>
                        </div>
                        <div className="result-item">
                            <span className="result-label">Processing Time</span>
                            <span className="result-value">
                                {result.processing_info?.processing_time_seconds?.toFixed(1)}s
                            </span>
                        </div>
                        <div className="result-item">
                            <span className="result-label">Effective FPS</span>
                            <span className="result-value">
                                {result.processing_info?.effective_fps?.toFixed(1)}
                            </span>
                        </div>
                    </div>

                    <hr />

                    <div className="result-grid">
                        <div className="result-item result-item--large">
                            <span className="result-label">Total Persons</span>
                            <span className="result-value result-value--highlight">
                                {result.aggregated_stats?.total_persons_detected}
                            </span>
                        </div>
                        <div className="result-item result-item--large">
                            <span className="result-label">Total Violations</span>
                            <span className="result-value result-value--violation">
                                {result.aggregated_stats?.total_violations}
                            </span>
                        </div>
                        <div className="result-item result-item--large">
                            <span className="result-label">Compliance Rate</span>
                            <span className={`result-value ${result.aggregated_stats?.compliance_rate > 80
                                    ? 'result-value--safe'
                                    : 'result-value--violation'
                                }`}>
                                {result.aggregated_stats?.compliance_rate?.toFixed(1)}%
                            </span>
                        </div>
                        <div className="result-item result-item--large">
                            <span className="result-label">Violation Frames</span>
                            <span className="result-value">
                                {result.aggregated_stats?.unique_violation_frames}
                            </span>
                        </div>
                    </div>

                    {result.output_video_url && (
                        <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                            <a
                                href={result.output_video_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="btn btn--primary"
                            >
                                📥 Download Annotated Video
                            </a>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default VideoUpload
