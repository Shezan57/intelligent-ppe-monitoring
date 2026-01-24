/**
 * UploadZone Component
 * Drag-and-drop file upload with preview
 */

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

const UploadZone = ({ onFileSelect, isLoading, selectedFile }) => {
    const [preview, setPreview] = useState(null)

    const onDrop = useCallback((acceptedFiles) => {
        const file = acceptedFiles[0]
        if (file) {
            // Create preview
            const reader = new FileReader()
            reader.onload = () => {
                setPreview(reader.result)
            }
            reader.readAsDataURL(file)

            // Notify parent
            onFileSelect(file)
        }
    }, [onFileSelect])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': ['.jpg', '.jpeg'],
            'image/png': ['.png'],
            'image/webp': ['.webp'],
        },
        maxSize: 10 * 1024 * 1024, // 10MB
        multiple: false,
        disabled: isLoading,
    })

    const clearSelection = (e) => {
        e.stopPropagation()
        setPreview(null)
        onFileSelect(null)
    }

    return (
        <div className="card">
            <div className="card__header">
                <h3 className="card__title">📤 Upload Image</h3>
                {selectedFile && (
                    <button
                        className="btn btn--secondary btn--icon"
                        onClick={clearSelection}
                        title="Clear selection"
                    >
                        ✕
                    </button>
                )}
            </div>

            <div
                {...getRootProps()}
                className={`upload-zone ${isDragActive ? 'upload-zone--active' : ''} ${selectedFile ? 'upload-zone--has-file' : ''}`}
            >
                <input {...getInputProps()} />

                {preview ? (
                    <div className="upload-zone__preview">
                        <img
                            src={preview}
                            alt="Preview"
                            style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px' }}
                        />
                        <p className="upload-zone__text mt-md">
                            {selectedFile?.name}
                        </p>
                        <span className="badge badge--safe">Ready to detect</span>
                    </div>
                ) : (
                    <>
                        <div className="upload-zone__icon">📷</div>
                        <p className="upload-zone__text">
                            {isDragActive
                                ? 'Drop the image here...'
                                : 'Drag & drop an image here, or click to select'
                            }
                        </p>
                        <p className="upload-zone__hint">
                            Supports: JPG, PNG, WebP (max 10MB)
                        </p>
                    </>
                )}
            </div>
        </div>
    )
}

export default UploadZone
