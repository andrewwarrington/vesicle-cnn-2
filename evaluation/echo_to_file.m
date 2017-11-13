function echo_to_file(string, file)

if exist('file')
	fileID = fopen(file,'a+');
	fprintf(fileID,strcat(string,'\n'));
	fclose(fileID);
end

disp(string)

end

