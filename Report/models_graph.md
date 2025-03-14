    <!-- https://mermaid.live/ -->
    gitGraph
       commit id: "GeoJSON input"
       commit id: "Segments"
       commit id: "Normalization"
       branch SegmentBasedClustering
       checkout SegmentBasedClustering
       commit id: "Geometric Preprocessing"
       checkout main
       branch VisionSegmentation
       checkout VisionSegmentation
       commit id: "Morph. closing & bluring"
       commit id: "Binary Image"
       branch SAM 
       checkout SAM
       commit id: "masks"
       checkout VisionSegmentation
       commit id: "cv2"
       merge SAM id: "np.array"
       checkout SegmentBasedClustering
       commit id: "closed loops"
       commit id: "merge smalls rooms"
       checkout VisionSegmentation
       commit id: "findcontours"
       checkout main
       commit id: "metric IoU"
       checkout VisionSegmentation
       checkout main
       merge SegmentBasedClustering
       merge VisionSegmentation id: "GeometryCollection"
       commit id: "IoU score"
       commit id: "GeoJSON export"