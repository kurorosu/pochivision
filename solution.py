def start_recording(self, *, name: str, quality: str = "High", output_directory: Optional[str] = None) -> bool:
        """
        Starts a new recording session.

        Parameters
        ----------
        name : str
            Name of the recording session.
        quality : str, optional
            Recording quality (`"Low"`, `"Medium"`, or `"High"`). Defaults to `"High"`.
        output_directory : str, optional
            Directory to store the recording. Defaults to the configured output directory.

        Returns
        -------
        bool
            True if recording started successfully, False if already recording.
        """
        with self.lock:
            if self.is_recording:
                return False

            # Resolve output directory
            output_dir = output_directory or self.output_directory
            if not output_dir:
                raise ValueError("Output directory must be specified.")

            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Prepare filename and full path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.mp4"
            self.output_file_path = os.path.join(output_dir, filename)

            # Initialize VideoWriter
            self.video_writer = self._initialize_video_writer(quality)

            # Set state
            self.is_recording = True
            self.start_time = time.time()
            return True