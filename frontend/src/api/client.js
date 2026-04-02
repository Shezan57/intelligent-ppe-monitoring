/**
 * API Client
 * Axios instance configured for backend API communication
 */

import axios from 'axios'

// Create axios instance
const api = axios.create({
    baseURL: '/api',
    timeout: 60000, // 60 seconds for detection
    headers: {
        'Content-Type': 'application/json',
    },
})

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        const message = error.response?.data?.detail || error.message || 'An error occurred'
        console.error('API Error:', message)
        return Promise.reject(error)
    }
)

/**
 * Run PPE detection on an uploaded image
 * @param {File} file - Image file to analyze
 * @param {Object} options - Optional parameters
 * @returns {Promise} Detection results
 */
export const detectViolations = async (file, options = {}) => {
    const formData = new FormData()
    formData.append('file', file)

    if (options.siteLocation) {
        formData.append('site_location', options.siteLocation)
    }
    if (options.cameraId) {
        formData.append('camera_id', options.cameraId)
    }
    formData.append('save_annotated', options.saveAnnotated !== false)

    const response = await api.post('/detect', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

/**
 * Upload an image file
 * @param {File} file - Image file to upload
 * @param {boolean} runDetection - Whether to run detection immediately
 * @returns {Promise} Upload result
 */
export const uploadImage = async (file, runDetection = false) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('run_detection', runDetection)

    const response = await api.post('/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

/**
 * Get violation history
 * @param {Object} params - Query parameters
 * @returns {Promise} History results
 */
export const getHistory = async (params = {}) => {
    const response = await api.get('/history', { params })
    return response.data
}

/**
 * Get history summary
 * @param {number} days - Number of days to summarize
 * @returns {Promise} Summary data
 */
export const getHistorySummary = async (days = 7) => {
    const response = await api.get('/history/summary', { params: { days } })
    return response.data
}

/**
 * Run Sentry-Judge pipeline on a video file
 * @param {File} file - Video file to process
 * @param {Object} options - Pipeline options
 * @returns {Promise} Combined Sentry + Judge results
 */
export const runPipeline = async (file, options = {}) => {
    const formData = new FormData()
    formData.append('file', file)
    if (options.cooldownSeconds) {
        formData.append('cooldown_seconds', options.cooldownSeconds)
    }
    if (options.cameraZone) {
        formData.append('camera_zone', options.cameraZone)
    }

    const response = await api.post('/detect/video/pipeline', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 600000, // 10 min for long videos
    })
    return response.data
}

/**
 * Get verified violations (Judge-confirmed only)
 * @param {Object} params - Query parameters
 * @returns {Promise} Verified violations list
 */
export const getVerifiedViolations = async (params = {}) => {
    const response = await api.get('/history/verified', { params })
    return response.data
}

/**
 * Check API health
 * @returns {Promise} Health status
 */
export const checkHealth = async () => {
    const response = await api.get('/health')
    return response.data
}

export default api
