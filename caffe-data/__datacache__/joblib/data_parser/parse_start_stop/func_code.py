# first line: 75
    @mem.cache
    def parse_start_stop(self, filepath=None):
        filename = self.filepath if self.filepath is not None else filepath
        if filename is None:
            # ``filename`` cannot be ``None``
            raise TypeError

        lines = []
        with open(filename, 'r') as in_file:
            start_stop_pattern = r"START|STOP"
            for line in in_file:
                # Make sure we only look at lines containing "START" or "STOP"
                if re.search(start_stop_pattern, line) is None:
                    continue

                # Create a tuple containing the timestamp, the label, and
                # whether it's a start or stop line
                split_line = line.strip().split()
                timestamp = split_line[1]
                start_stop = split_line[-1]
                label = split_line[-2]
                lines.append((timestamp, label, start_stop))

        return lines
