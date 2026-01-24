/**
 * DetectionCanvas Component
 * Displays annotated detection image with bounding boxes
 */

import { useState, useRef, useEffect } from 'react'

const DetectionCanvas = ({
    originalImage,
    annotatedImage,
    detectionResult,
    isLoading
}) => {
    const [showAnnotated, setShowAnnotated] = useState(true)
    const canvasRef = useRef(null)

    // Get the image URL to display
    const displayImage = showAnnotated && annotatedImage
        ? annotatedImage
        : originalImage

    // Draw bounding boxes on canvas (if we want custom rendering)
    useEffect(() => {
        if (!canvasRef.current || !originalImage || !detectionResult?.persons) return

        // For now, we use the backend-annotated image
        // This effect is for future custom overlays
    }, [originalImage, detectionResult])

    if (isLoading) {
        return (
            <div className="card detection-canvas">
                <div className="loading-overlay" style={{ position: 'relative', background: 'transparent' }}>
                    <div className="loading-spinner"></div>
                    <p className="loading-overlay__text">Analyzing image with hybrid detection...</p>
                </div>
            </div>
        )
    }

    if (!displayImage) {
        return (
            <div className="card detection-canvas">
                <div className="detection-canvas__placeholder">
                    <div style={{ fontSize: '4rem', marginBottom: '1rem', opacity: 0.3 }}>🔍</div>
                    <h3>No Image Selected</h3>
                    <p>Upload an image to run PPE detection</p>
                </div>
            </div>
        )
    }

    return (
        <div className="card">
            <div className="card__header">
                <h3 className="card__title">🖼️ Detection Result</h3>
                {annotatedImage && originalImage && (
                    <div className="flex gap-sm">
                        <button
                            className={`btn ${!showAnnotated ? 'btn--primary' : 'btn--secondary'}`}
                            onClick={() => setShowAnnotated(false)}
                        >
                            Original
                        </button>
                        <button
                            className={`btn ${showAnnotated ? 'btn--primary' : 'btn--secondary'}`}
                            onClick={() => setShowAnnotated(true)}
                        >
                            Annotated
                        </button>
                    </div>
                )}
            </div>

            <div className="detection-canvas">
                <img
                    src={displayImage}
                    alt="Detection result"
                    className="detection-canvas__image"
                    style={{ width: '100%', height: 'auto' }}
                />

                {/* Legend overlay */}
                {detectionResult && (
                    <div style={{
                        position: 'absolute',
                        bottom: '1rem',
                        right: '1rem',
                        background: 'rgba(15, 23, 42, 0.9)',
                        padding: '0.75rem 1rem',
                        borderRadius: '0.5rem',
                        fontSize: '0.75rem',
                        display: 'flex',
                        gap: '1rem',
                    }}>
                        <div className="flex items-center gap-sm">
                            <span style={{
                                display: 'inline-block',
                                width: '12px',
                                height: '12px',
                                background: '#10b981',
                                borderRadius: '2px'
                            }}></span>
                            <span>Safe</span>
                        </div>
                        <div className="flex items-center gap-sm">
                            <span style={{
                                display: 'inline-block',
                                width: '12px',
                                height: '12px',
                                background: '#ef4444',
                                borderRadius: '2px'
                            }}></span>
                            <span>Violation</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Detection summary */}
            {detectionResult?.stats && (
                <div style={{
                    marginTop: '1rem',
                    padding: '0.75rem',
                    background: 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-md)',
                    fontSize: '0.875rem'
                }}>
                    <div className="flex justify-between">
                        <span>Processing Time:</span>
                        <strong>{detectionResult.timing?.total_ms.toFixed(1)}ms</strong>
                    </div>
                    <div className="flex justify-between mt-sm">
                        <span>SAM Bypass Rate:</span>
                        <strong>{detectionResult.stats.bypass_rate.toFixed(1)}%</strong>
                    </div>
                </div>
            )}
        </div>
    )
}

export default DetectionCanvas
