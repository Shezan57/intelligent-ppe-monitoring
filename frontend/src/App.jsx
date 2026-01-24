/**
 * PPE Detection Frontend - Main Application
 * 
 * Intelligent PPE Compliance Monitoring System
 * Master's Thesis Project
 */

import { useState, useCallback } from 'react'
import toast from 'react-hot-toast'

import Header from './components/Header'
import UploadZone from './components/UploadZone'
import DetectionCanvas from './components/DetectionCanvas'
import StatsPanel from './components/StatsPanel'
import ViolationCard from './components/ViolationCard'
import { detectViolations } from './api/client'

function App() {
    // State
    const [selectedFile, setSelectedFile] = useState(null)
    const [originalImage, setOriginalImage] = useState(null)
    const [annotatedImage, setAnnotatedImage] = useState(null)
    const [detectionResult, setDetectionResult] = useState(null)
    const [isLoading, setIsLoading] = useState(false)

    // Handle file selection
    const handleFileSelect = useCallback((file) => {
        setSelectedFile(file)
        setDetectionResult(null)
        setAnnotatedImage(null)

        if (file) {
            // Create preview URL for original image
            const url = URL.createObjectURL(file)
            setOriginalImage(url)
        } else {
            setOriginalImage(null)
        }
    }, [])

    // Run detection
    const handleDetect = useCallback(async () => {
        if (!selectedFile) {
            toast.error('Please select an image first')
            return
        }

        setIsLoading(true)
        const loadingToast = toast.loading('Analyzing image with hybrid detection...')

        try {
            const result = await detectViolations(selectedFile, {
                saveAnnotated: true,
            })

            setDetectionResult(result)

            // Set annotated image URL
            if (result.annotated_image_path) {
                // Convert server path to URL
                const filename = result.annotated_image_path.split(/[/\\]/).pop()
                setAnnotatedImage(`/uploads/${filename}`)
            }

            // Show result toast
            const violations = result.stats?.total_violations || 0
            const total = result.stats?.total_persons || 0

            if (violations === 0 && total > 0) {
                toast.success(`✅ All ${total} workers are compliant!`, { id: loadingToast })
            } else if (violations > 0) {
                toast.error(`⚠️ ${violations} violation(s) detected out of ${total} workers`, { id: loadingToast })
            } else {
                toast.success('Detection completed', { id: loadingToast })
            }

        } catch (error) {
            console.error('Detection error:', error)
            toast.error(
                error.response?.data?.detail || 'Detection failed. Is the backend running?',
                { id: loadingToast }
            )
        } finally {
            setIsLoading(false)
        }
    }, [selectedFile])

    // Reset all state
    const handleReset = useCallback(() => {
        setSelectedFile(null)
        setOriginalImage(null)
        setAnnotatedImage(null)
        setDetectionResult(null)
    }, [])

    return (
        <>
            <Header />

            <main className="container">
                <div className="app-layout">
                    {/* Sidebar */}
                    <aside className="sidebar">
                        {/* Upload Zone */}
                        <UploadZone
                            onFileSelect={handleFileSelect}
                            isLoading={isLoading}
                            selectedFile={selectedFile}
                        />

                        {/* Action Buttons */}
                        <div className="flex gap-md">
                            <button
                                className="btn btn--primary btn--lg"
                                style={{ flex: 1 }}
                                onClick={handleDetect}
                                disabled={!selectedFile || isLoading}
                            >
                                {isLoading ? (
                                    <>
                                        <span className="loading-spinner" style={{ width: '16px', height: '16px' }}></span>
                                        Detecting...
                                    </>
                                ) : (
                                    <>🔍 Detect PPE</>
                                )}
                            </button>

                            {detectionResult && (
                                <button
                                    className="btn btn--secondary btn--lg"
                                    onClick={handleReset}
                                >
                                    Clear
                                </button>
                            )}
                        </div>

                        {/* Stats Panel */}
                        <StatsPanel
                            stats={detectionResult?.stats}
                            timing={detectionResult?.timing}
                        />

                        {/* System Info */}
                        <div className="card" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                                ℹ️ System Info
                            </h4>
                            <p>• Hybrid YOLO + SAM detection</p>
                            <p>• 5-path intelligent bypass</p>
                            <p>• 79.8% SAM bypass rate</p>
                            <p>• Real-time performance ~28 FPS</p>
                        </div>
                    </aside>

                    {/* Main Content */}
                    <div className="main-content">
                        {/* Detection Canvas */}
                        <DetectionCanvas
                            originalImage={originalImage}
                            annotatedImage={annotatedImage}
                            detectionResult={detectionResult}
                            isLoading={isLoading}
                        />

                        {/* Violation Cards */}
                        {detectionResult?.persons && detectionResult.persons.length > 0 && (
                            <div className="results-section">
                                <div className="results-section__header">
                                    <h3>👷 Detected Workers ({detectionResult.persons.length})</h3>
                                    <div className="flex gap-sm">
                                        <span className="badge badge--safe">
                                            {detectionResult.persons.filter(p => p.has_helmet && p.has_vest).length} Safe
                                        </span>
                                        <span className="badge badge--violation">
                                            {detectionResult.persons.filter(p => !(p.has_helmet && p.has_vest)).length} Violations
                                        </span>
                                    </div>
                                </div>

                                <div className="violation-list">
                                    {detectionResult.persons.map((person) => (
                                        <ViolationCard key={person.person_id} person={person} />
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Empty State */}
                        {!detectionResult && !isLoading && (
                            <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                                <div style={{ fontSize: '4rem', marginBottom: '1rem', opacity: 0.3 }}>🏗️</div>
                                <h3>PPE Compliance Monitoring</h3>
                                <p style={{ maxWidth: '500px', margin: '1rem auto' }}>
                                    Upload a construction site image to detect workers and verify
                                    their Personal Protective Equipment (helmet and vest) compliance.
                                </p>
                                <div className="flex gap-md justify-center mt-lg" style={{ justifyContent: 'center' }}>
                                    <div style={{ textAlign: 'center' }}>
                                        <div style={{ fontSize: '2rem' }}>⛑️</div>
                                        <small>Helmet</small>
                                    </div>
                                    <div style={{ textAlign: 'center' }}>
                                        <div style={{ fontSize: '2rem' }}>🦺</div>
                                        <small>Vest</small>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer style={{
                textAlign: 'center',
                padding: '1rem',
                color: 'var(--text-muted)',
                fontSize: '0.75rem',
                borderTop: '1px solid var(--border-color)',
                marginTop: 'auto'
            }}>
                Master's Thesis Project • Intelligent PPE Compliance Monitoring •
                Hybrid YOLO + SAM Detection with 5-Path Intelligent Bypass
            </footer>
        </>
    )
}

export default App
