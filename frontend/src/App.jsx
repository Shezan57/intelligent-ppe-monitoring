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
import HistoryTable from './components/HistoryTable'
import SettingsPanel from './components/SettingsPanel'
import VideoUpload from './components/VideoUpload'
import { detectViolations } from './api/client'

function App() {
    // Navigation
    const [activeTab, setActiveTab] = useState('image')
    const [showSettings, setShowSettings] = useState(false)

    // Image Detection State
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

            if (result.annotated_image_path) {
                const filename = result.annotated_image_path.split(/[/\\]/).pop()
                setAnnotatedImage(`/uploads/${filename}`)
            }

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
            <Header onSettingsClick={() => setShowSettings(true)} />

            <main className="container">
                {/* Navigation Tabs */}
                <div className="nav-tabs" style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
                    <button
                        className={`nav-tab ${activeTab === 'image' ? 'nav-tab--active' : ''}`}
                        onClick={() => setActiveTab('image')}
                    >
                        📷 Image Detection
                    </button>
                    <button
                        className={`nav-tab ${activeTab === 'video' ? 'nav-tab--active' : ''}`}
                        onClick={() => setActiveTab('video')}
                    >
                        🎬 Video Detection
                    </button>
                    <button
                        className={`nav-tab ${activeTab === 'history' ? 'nav-tab--active' : ''}`}
                        onClick={() => setActiveTab('history')}
                    >
                        📋 Violation History
                    </button>
                </div>

                {/* Image Detection Tab */}
                {activeTab === 'image' && (
                    <div className="app-layout">
                        {/* Sidebar */}
                        <aside className="sidebar">
                            <UploadZone
                                onFileSelect={handleFileSelect}
                                isLoading={isLoading}
                                selectedFile={selectedFile}
                            />

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

                            <StatsPanel
                                stats={detectionResult?.stats}
                                timing={detectionResult?.timing}
                            />

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
                            <DetectionCanvas
                                originalImage={originalImage}
                                annotatedImage={annotatedImage}
                                detectionResult={detectionResult}
                                isLoading={isLoading}
                            />

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
                )}

                {/* Video Detection Tab */}
                {activeTab === 'video' && (
                    <div className="app-layout">
                        <aside className="sidebar">
                            <VideoUpload />
                        </aside>
                        <div className="main-content">
                            <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                                <div style={{ fontSize: '4rem', marginBottom: '1rem', opacity: 0.3 }}>🎬</div>
                                <h3>Video PPE Detection</h3>
                                <p style={{ maxWidth: '500px', margin: '1rem auto' }}>
                                    Upload a video file to analyze PPE compliance across all frames.
                                    The system will process the video and provide aggregate statistics.
                                </p>
                                <div className="flex gap-md" style={{ justifyContent: 'center', marginTop: '1.5rem' }}>
                                    <div className="card" style={{ padding: '1rem', flex: 1, maxWidth: '150px' }}>
                                        <div style={{ fontSize: '1.5rem' }}>📹</div>
                                        <small>MP4, AVI, MOV</small>
                                    </div>
                                    <div className="card" style={{ padding: '1rem', flex: 1, maxWidth: '150px' }}>
                                        <div style={{ fontSize: '1.5rem' }}>⚡</div>
                                        <small>Frame Skip</small>
                                    </div>
                                    <div className="card" style={{ padding: '1rem', flex: 1, maxWidth: '150px' }}>
                                        <div style={{ fontSize: '1.5rem' }}>📊</div>
                                        <small>Aggregate Stats</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* History Tab */}
                {activeTab === 'history' && (
                    <HistoryTable />
                )}
            </main>

            {/* Settings Modal */}
            {showSettings && (
                <SettingsPanel onClose={() => setShowSettings(false)} />
            )}

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
